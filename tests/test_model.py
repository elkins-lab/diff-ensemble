import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays

from diff_ensemble.model import Decoder, Encoder, build_backbone_coords

# ---------------------------------------------------------------------------
# build_backbone_coords — shape contract and coordinate correctness
# ---------------------------------------------------------------------------


class TestBuildBackboneCoords:
    @pytest.mark.parametrize("ensemble_size", [1, 7, 10])
    @pytest.mark.parametrize("seq_len", [2, 12, 50])
    def test_output_shape(self, ensemble_size, seq_len):
        """Output must be (ensemble_size, seq_len * 3, 3)."""
        torsions = jnp.zeros((ensemble_size, seq_len, 2))
        coords = build_backbone_coords(torsions)
        assert coords.shape == (ensemble_size, seq_len * 3, 3)

    @pytest.mark.parametrize("ensemble_size, seq_len", [(3, 5), (1, 10)])
    def test_first_atom_at_origin(self, ensemble_size, seq_len):
        """The first atom (N of residue 1) is always placed at the origin."""
        torsions = jnp.zeros((ensemble_size, seq_len, 2))
        coords = build_backbone_coords(torsions)
        # First atom of every model should be (0, 0, 0)
        assert bool(jnp.allclose(coords[:, 0, :], jnp.zeros((ensemble_size, 3)), atol=1e-5))

    @settings(deadline=None)
    @given(arrays(np.float32, (3, 10, 2), elements=st.floats(min_value=-np.pi, max_value=np.pi)))
    def test_coordinates_are_finite(self, np_torsions):
        """Random torsions should never produce NaN or Inf coordinates."""
        torsions = jnp.array(np_torsions)
        coords = build_backbone_coords(torsions)
        assert bool(jnp.all(jnp.isfinite(coords)))

    def test_different_torsions_give_different_coords(self):
        """Two distinct torsion sets must produce distinct coordinate sets."""
        torsions_a = jnp.zeros((1, 5, 2))
        torsions_b = jnp.ones((1, 5, 2)) * 0.5
        coords_a = build_backbone_coords(torsions_a)
        coords_b = build_backbone_coords(torsions_b)
        assert not bool(jnp.allclose(coords_a, coords_b))

    def test_invalid_torsions_shape(self):
        """Ensure exception is raised for invalid torsion shapes."""
        with pytest.raises(ValueError, match="Expected torsions shape"):
            build_backbone_coords(jnp.zeros((10, 20)))  # Missing dimension


# ---------------------------------------------------------------------------
# Encoder and Decoder — in isolation
# ---------------------------------------------------------------------------


class TestEncoder:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [5, 10])
    @pytest.mark.parametrize("features", [2, 4])
    @pytest.mark.parametrize("latent_dim", [8, 16])
    def test_output_shapes(self, batch_size, seq_len, features, latent_dim):
        """Encoder must return two tensors both shaped (batch_size, latent_dim)."""
        hidden_dim = 32

        encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((batch_size, seq_len, features))
        params = encoder.init(rng, x)["params"]  # type: ignore[index]
        mean, logvar = encoder.apply({"params": params}, x)  # type: ignore[misc]

        assert mean.shape == (batch_size, latent_dim)  # type: ignore[union-attr]
        assert logvar.shape == (batch_size, latent_dim)  # type: ignore[union-attr]

    def test_mean_and_logvar_are_different(self):
        """Mean and log-variance heads should produce different values."""
        encoder = Encoder(latent_dim=8, hidden_dim=32)
        rng = jax.random.PRNGKey(1)
        x = jnp.ones((1, 5, 4))
        params = encoder.init(rng, x)["params"]  # type: ignore[index]
        mean, logvar = encoder.apply({"params": params}, x)  # type: ignore[misc]
        assert not bool(jnp.allclose(mean, logvar))  # type: ignore[arg-type]

    def test_invalid_input_shape(self):
        """Ensure exception is raised for invalid x shapes."""
        encoder = Encoder(latent_dim=8, hidden_dim=32)
        rng = jax.random.PRNGKey(0)
        x_init = jnp.ones((1, 5, 4))
        params = encoder.init(rng, x_init)["params"]  # type: ignore[index]

        with pytest.raises(ValueError, match="Expected x shape"):
            encoder.apply({"params": params}, jnp.ones((5, 4)))  # Missing batch dim


class TestDecoder:
    @pytest.mark.parametrize("ensemble_size", [1, 8])
    @pytest.mark.parametrize("seq_len", [5, 10])
    @pytest.mark.parametrize("latent_dim", [8, 16])
    def test_output_shape(self, ensemble_size, seq_len, latent_dim):
        """Decoder must return (ensemble_size, seq_len, 2) torsion tensor."""
        decoder = Decoder(seq_len=seq_len, hidden_dim=32)
        rng = jax.random.PRNGKey(0)
        z = jnp.ones((ensemble_size, latent_dim))
        params = decoder.init(rng, z)["params"]  # type: ignore[index]
        torsions = decoder.apply({"params": params}, z)  # type: ignore[misc]
        assert torsions.shape == (ensemble_size, seq_len, 2)  # type: ignore[union-attr]

    @settings(deadline=None)
    @given(arrays(np.float32, (10, 16), elements=st.floats(min_value=-100.0, max_value=100.0)))
    def test_torsions_bounded_in_pi(self, np_z):
        """tanh * π activation keeps all torsions in the closed interval [−π, π]."""
        seq_len = 10
        decoder = Decoder(seq_len=seq_len, hidden_dim=64)
        rng = jax.random.PRNGKey(2)
        z = jnp.array(np_z)
        params = decoder.init(rng, z)["params"]  # type: ignore[index]
        torsions = decoder.apply({"params": params}, z)  # type: ignore[misc]
        assert bool(jnp.all(torsions >= -jnp.pi))  # type: ignore[operator]
        assert bool(jnp.all(torsions <= jnp.pi))  # type: ignore[operator]

    def test_invalid_input_shape(self):
        """Ensure exception is raised for invalid z shapes."""
        decoder = Decoder(seq_len=10, hidden_dim=32)
        rng = jax.random.PRNGKey(0)
        z_init = jnp.ones((1, 16))
        params = decoder.init(rng, z_init)["params"]  # type: ignore[index]

        with pytest.raises(ValueError, match="Expected z shape"):
            decoder.apply({"params": params}, jnp.ones((16,)))  # Missing ensemble dim
