"""Coleman's random placement model: the formulae, the verdict and input checks.
Agreement with R is in test_r_reference.py."""

import warnings

import numpy as np
import pandas as pd
import pytest

import sars
from sars import ColemanFit, ColemanTest, coleman


def _simulate(seed: int, n_species: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """A community placed at random over eight sites."""
    rng = np.random.default_rng(seed)
    area = np.array([1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0, 120.0])
    totals = rng.geometric(0.05, size=n_species)
    counts = np.array([rng.multinomial(n, area / area.sum()) for n in totals])
    return counts, area


def _confined() -> tuple[np.ndarray, np.ndarray]:
    """Twelve species, each kept to a single site: far fewer species per site
    than random placement would give."""
    counts = np.zeros((12, 6), dtype=int)
    counts[np.arange(12), np.arange(12) % 6] = 20
    return counts, np.arange(1.0, 7.0)


class TestFormulae:
    def test_expected_and_sd_by_hand(self):
        # Species totals 2 and 3; two sites holding 0.25 and 0.75 of the area.
        fit = coleman([[2, 0], [1, 2]], [1.0, 3.0])
        assert isinstance(fit, ColemanFit)
        np.testing.assert_allclose(fit.sites["relative_area"], [0.25, 0.75])
        np.testing.assert_allclose(fit.abundance, [2, 3])
        # Chance each species is absent from the first site, then the second.
        p1 = np.array([0.75**2, 0.75**3])
        p2 = np.array([0.25**2, 0.25**3])
        np.testing.assert_allclose(fit.sites["expected"], [(1 - p1).sum(), (1 - p2).sum()])
        np.testing.assert_allclose(
            fit.sites["sd"],
            [np.sqrt(p1.sum() - (p1**2).sum()), np.sqrt(p2.sum() - (p2**2).sum())],
        )
        assert fit.sites["observed"].tolist() == [2, 1]
        assert fit.total_area == 4.0
        assert fit.n_sites == 2
        assert fit.n_species == 2

    def test_sites_table_columns_and_order(self):
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        assert list(fit.sites.columns) == [
            "area",
            "relative_area",
            "observed",
            "expected",
            "sd",
            "within_sd",
        ]
        np.testing.assert_allclose(fit.sites["area"], area)
        assert fit.sites["relative_area"].sum() == pytest.approx(1.0)
        assert fit.sites.index.name == "site"
        assert fit.sites.index.tolist() == list(range(len(area)))

    def test_simulated_random_placement_has_the_models_mean_and_sd(self):
        """Over many random placements of the same species totals, the richness
        of each site averages the expected value with the model's spread."""
        rng = np.random.default_rng(0)
        area = np.array([1.0, 3.0, 10.0, 40.0])
        totals = np.array([1, 2, 4, 7, 12, 20, 33, 60])
        # Shape (runs, species, sites).
        placed = rng.multinomial(totals, area / area.sum(), size=(20000, len(totals)))
        richness = (placed > 0).sum(axis=1)
        fit = coleman(placed[0], area)
        np.testing.assert_allclose(richness.mean(axis=0), fit.sites["expected"], atol=0.03)
        np.testing.assert_allclose(richness.std(axis=0), fit.sites["sd"], rtol=0.03)

    def test_predict_reproduces_the_site_table(self):
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        np.testing.assert_allclose(fit.predict(area), fit.sites["expected"])
        np.testing.assert_allclose(fit.predict_sd(area), fit.sites["sd"])

    def test_predict_at_the_ends_of_the_area_range(self):
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        # No area holds no species; the whole area holds all of them, for certain.
        assert fit.predict(0.0) == pytest.approx(0.0)
        assert fit.predict(fit.total_area) == pytest.approx(fit.n_species)
        assert fit.predict_sd(0.0) == pytest.approx(0.0)
        assert fit.predict_sd(fit.total_area) == pytest.approx(0.0)

    def test_predict_accepts_the_total_area_summed_another_way(self):
        """Summing the same areas in another order can land one step above
        ``total_area``; that is still the whole area, not an area outside it."""
        area = [0.1, 0.2, 0.3, 0.7, 1.1, 2.3]
        fit = coleman(np.full((3, 6), 4), area)
        above = np.nextafter(fit.total_area, np.inf)
        assert fit.predict(above) == pytest.approx(fit.n_species)
        assert fit.predict_sd(above) == pytest.approx(0.0)
        assert fit.predict(sum(reversed(area))) == pytest.approx(fit.n_species)
        with pytest.raises(ValueError, match="between 0 and the total area"):
            fit.predict(fit.total_area * 1.001)

    def test_predict_keeps_the_shape_of_its_input(self):
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        assert fit.predict(5.0).shape == ()
        assert fit.predict([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
        assert fit.predict_sd([1.0, 2.0, 3.0]).shape == (3,)

    @pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf, 1e6])
    def test_predict_rejects_areas_outside_the_total(self, bad):
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        with pytest.raises(ValueError, match="between 0 and the total area"):
            fit.predict(bad)
        with pytest.raises(ValueError, match="between 0 and the total area"):
            fit.predict_sd([1.0, bad])

    def test_large_abundances_stay_finite(self):
        fit = coleman([[10**6, 10**7], [1, 0], [3, 5]], [1.0, 1e6])
        assert np.all(np.isfinite(fit.sites[["expected", "sd"]].to_numpy()))
        assert np.all(fit.sites["sd"] >= 0)


class TestVerdict:
    def test_within_sd_and_fraction_outside(self):
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        s = fit.sites
        expected_within = (s["observed"] - s["expected"]).abs() <= s["sd"]
        assert s["within_sd"].tolist() == expected_within.tolist()
        assert s["within_sd"].dtype == bool
        assert fit.fraction_outside == pytest.approx(1 / 8)
        assert fit.rejected is False

    def test_rejected_when_species_keep_to_one_site(self):
        fit = coleman(*_confined())
        assert not fit.sites["within_sd"].any()
        assert fit.fraction_outside == 1.0
        assert fit.rejected is True

    @pytest.mark.parametrize(
        ("outside", "rejected"),
        [(0, False), (1, False), (2, False), (3, True), (6, True)],
    )
    def test_threshold_is_more_than_a_third(self, outside, rejected):
        """Exactly a third of the sites outside (2 of 6) doesn't reject."""
        within = np.array([False] * outside + [True] * (6 - outside))
        fit = ColemanFit(
            sites=pd.DataFrame({"within_sd": within}),
            abundance=pd.Series([1.0]),
            total_area=1.0,
        )
        assert fit.fraction_outside == pytest.approx(outside / 6)
        assert fit.rejected is rejected

    def test_fraction_outside_follows_the_site_table(self):
        """The share, the verdict and the repr are read from ``sites``, so they
        can't disagree with it."""
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        fit.sites["within_sd"] = False
        assert fit.fraction_outside == 1.0
        assert fit.rejected is True
        assert "8 outside ±1 sd (100%), random placement rejected" in repr(fit)

    def test_site_on_the_edge_of_the_band_counts_as_within(self):
        # With thousands of individuals each species is at both sites for
        # certain: expected equals observed, sd is 0, and each site sits
        # exactly on the edge of a band of no width.
        fit = coleman([[5000, 5000], [4000, 6000]], [1.0, 1.0])
        np.testing.assert_array_equal(fit.sites["sd"], [0.0, 0.0])
        assert fit.sites["within_sd"].all()
        assert fit.fraction_outside == 0.0

    def test_repr(self):
        counts, area = _simulate(3)
        assert repr(coleman(counts, area)) == (
            "ColemanFit(30 species, 8 sites, 1 outside ±1 sd (12%), random placement not rejected)"
        )
        assert repr(coleman(*_confined())) == (
            "ColemanFit(12 species, 6 sites, 6 outside ±1 sd (100%), random placement rejected)"
        )


class TestPlacementTest:
    def test_calibrated_on_random_placement(self):
        """On communities placed at random, the test rejects about as often as its
        level says, where the one-third rule rejects about half of them."""
        rng = np.random.default_rng(5)
        p_values, rule = [], []
        for seed in range(300):
            fit = coleman(*_simulate(seed))
            p_values.append(fit.placement_test(n_sim=199, rng=rng).p_value)
            rule.append(fit.rejected)
        p_values = np.array(p_values)
        assert 0.01 <= np.mean(p_values <= 0.05) <= 0.10
        assert 0.12 <= np.mean(p_values <= 0.20) <= 0.28
        assert np.mean(rule) > 0.3

    def test_species_kept_to_one_site_are_rejected(self):
        result = coleman(*_confined()).placement_test(n_sim=99, rng=np.random.default_rng(0))
        assert result.p_value == 0.01  # the smallest possible: 1 / (99 + 1)

    def test_statistic_and_rule_rate(self):
        fit = coleman(*_simulate(1))
        result = fit.placement_test(n_sim=500, rng=np.random.default_rng(2))
        sites = fit.sites
        z = (sites["observed"] - sites["expected"]) / sites["sd"]
        assert result.statistic == pytest.approx(float((z**2).sum()))
        assert 0.0 <= result.rule_rejection_rate <= 1.0
        assert result.n_sim == 500
        assert 1 / 501 <= result.p_value <= 1.0

    def test_reproducible_with_a_seed(self):
        fit = coleman(*_simulate(3))
        first = fit.placement_test(n_sim=99, rng=np.random.default_rng(7))
        second = fit.placement_test(n_sim=99, rng=np.random.default_rng(7))
        assert first == second
        assert fit.placement_test(n_sim=99, rng=7) == first  # a seed works too

    def test_simulations_that_tie_with_the_data_count(self):
        """Species so abundant that every site holds all of them, in the data and
        in every simulation: each simulation ties, so p is 1."""
        counts = np.full((5, 4), 500)
        result = coleman(counts, np.array([1.0, 2.0, 3.0, 4.0])).placement_test(
            n_sim=50, rng=np.random.default_rng(0)
        )
        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_a_site_that_random_placement_cant_produce(self):
        """A site whose richness is fixed under random placement (sd 0) makes the
        distance infinite when it doesn't match, and adds nothing when it does."""
        from sars._coleman import _distance

        expected, sd = np.array([3.0, 5.0]), np.array([1.0, 0.0])
        assert _distance(np.array([4.0, 5.0]), expected, sd) == 1.0
        assert _distance(np.array([4.0, 4.0]), expected, sd) == np.inf

    @pytest.mark.parametrize("n_sim", [0, -5, 2.5, True, "99"])
    def test_bad_n_sim(self, n_sim):
        with pytest.raises(ValueError, match="n_sim must be a positive whole number"):
            coleman(*_simulate(1)).placement_test(n_sim=n_sim)

    def test_repr(self):
        result = coleman(*_confined()).placement_test(n_sim=9, rng=np.random.default_rng(0))
        assert repr(result).startswith("ColemanTest(statistic=")


class TestInputs:
    def test_dataframe_labels_become_the_site_index(self):
        counts, area = _simulate(3)
        names = [f"site{i + 1}" for i in range(len(area))]
        df = pd.DataFrame(counts, columns=names, index=[f"sp{i}" for i in range(len(counts))])
        fit = coleman(df, area)
        assert fit.sites.index.tolist() == names
        np.testing.assert_allclose(fit.sites["expected"], coleman(counts, area).sites["expected"])
        np.testing.assert_array_equal(
            fit.sites["observed"], coleman(counts, area).sites["observed"]
        )

    def test_labelled_area_is_matched_to_the_columns_by_label(self):
        counts, area = _simulate(3)
        names = list("abcdefgh")
        df = pd.DataFrame(counts, columns=names)
        by_label = pd.Series(area, index=names)
        # The same areas listed in another order still go to their own sites.
        fit = coleman(df, by_label.iloc[::-1])
        assert fit.sites.index.tolist() == names
        np.testing.assert_allclose(fit.sites["area"], area)
        pd.testing.assert_frame_equal(fit.sites, coleman(df, area).sites)
        pd.testing.assert_frame_equal(fit.sites, coleman(df, by_label).sites)

    @pytest.mark.parametrize(
        "index",
        [
            list("abcdefgx"),  # one name differs
            list("ijklmnop"),  # none in common
            list("aabcdefg"),  # a name twice, so a site is left without an area
        ],
    )
    def test_labelled_area_with_other_labels_is_rejected(self, index):
        counts, area = _simulate(3)
        df = pd.DataFrame(counts, columns=list("abcdefgh"))
        with pytest.raises(ValueError, match="labels don't match"):
            coleman(df, pd.Series(area, index=index))

    def test_labelled_area_of_the_wrong_length_reports_the_length(self):
        counts, area = _simulate(3)
        df = pd.DataFrame(counts, columns=list("abcdefgh"))
        with pytest.raises(ValueError, match="one value per site"):
            coleman(df, pd.Series(area[:5], index=list("abcde")))

    def test_unlabelled_input_is_matched_by_position(self):
        counts, area = _simulate(3)
        df = pd.DataFrame(counts, columns=list("abcdefgh"))
        expected = coleman(counts, area).sites["area"].to_numpy()
        labelled = pd.Series(area, index=list("hgfedcba"))
        # A Series with pandas's default labels, or a list, against named columns.
        np.testing.assert_allclose(coleman(df, pd.Series(area)).sites["area"], expected)
        np.testing.assert_allclose(coleman(df, list(area)).sites["area"], expected)
        # A labelled Series against a table with no site names: an array, or a
        # DataFrame with default column labels.
        np.testing.assert_allclose(coleman(counts, labelled).sites["area"], expected)
        np.testing.assert_allclose(coleman(pd.DataFrame(counts), labelled).sites["area"], expected)

    def test_abundance_totals_keep_their_species_labels(self):
        df = pd.DataFrame(
            [[5, 0, 1], [0, 0, 0], [1, 1, 0], [0, 0, 0], [3, 0, 0]],
            index=["wren", "absent1", "robin", "absent2", "finch"],
            columns=["x", "y", "z"],
        )
        fit = coleman(df, [1.0, 2.0, 3.0])
        assert isinstance(fit.abundance, pd.Series)
        assert fit.abundance.to_dict() == {"wren": 6, "robin": 2, "finch": 3}
        assert fit.abundance.index.name == "species"
        # An array's species are known by their row positions.
        assert coleman(df.to_numpy(), [1.0, 2.0, 3.0]).abundance.index.tolist() == [0, 2, 4]

    def test_fits_compare_by_identity(self):
        """Comparing the tables field by field has no single answer, so two fits
        are equal only when they are the same object."""
        counts, area = _simulate(3)
        fit = coleman(counts, area)
        assert fit == fit
        assert fit != coleman(counts, area)
        assert fit in [fit]

    def test_rows_are_species_and_columns_sites(self):
        counts = np.array([[5, 0, 1], [0, 2, 2], [1, 1, 0], [3, 0, 0]])  # 4 species, 3 sites
        fit = coleman(counts, [1.0, 2.0, 3.0])
        assert fit.n_species == 4
        assert fit.n_sites == 3
        assert fit.sites["observed"].tolist() == [3, 2, 2]
        np.testing.assert_allclose(fit.abundance, [6, 4, 2, 3])
        with pytest.raises(ValueError, match="one value per site"):
            coleman(counts.T, [1.0, 2.0, 3.0])

    def test_species_absent_everywhere_are_dropped_without_changing_the_result(self):
        counts, area = _simulate(3)
        padded = np.vstack([counts, np.zeros((4, counts.shape[1]), dtype=int)])
        fit = coleman(padded, area)
        assert fit.n_species == coleman(counts, area).n_species
        pd.testing.assert_frame_equal(fit.sites, coleman(counts, area).sites)

    def test_whole_numbers_stored_as_floats_are_accepted(self):
        counts, area = _simulate(3)
        pd.testing.assert_frame_equal(
            coleman(counts.astype(float), area).sites, coleman(counts, area).sites
        )

    def test_no_warning_for_abundance_data(self):
        counts, area = _simulate(3)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            coleman(counts, area)

    def test_presence_absence_data_warns(self):
        counts, area = _simulate(3)
        with pytest.warns(UserWarning, match="presence-absence"):
            coleman((counts > 0).astype(int), area)

    @pytest.mark.parametrize(
        ("counts", "match"),
        [
            ([1, 2, 3], "two-dimensional"),
            ([[[1, 2]]], "two-dimensional"),
            ([[1], [2]], "at least 2 sites"),
            ([[1, np.nan], [2, 3]], "missing or non-finite"),
            ([[1, np.inf], [2, 3]], "missing or non-finite"),
            ([[1, -1], [2, 3]], "negative"),
            ([[1, 0.5], [2, 3]], "fractional"),
            ([[0, 0], [0, 0]], "no individuals"),
            ([[1, None], [2, 3]], "missing or non-finite"),
            ([["a", "b"], ["c", "d"]], "numbers only"),
            # Numbers stored as text are refused too, not read as counts.
            ([["1", "2"], ["3", "4"]], "numbers only"),
            ([[1, "2"], [3, 4]], "numbers only"),
            ([[1, 2], [3]], "rectangular"),
        ],
    )
    def test_bad_abundance(self, counts, match):
        with pytest.raises(ValueError, match=match):
            coleman(counts, [1.0, 2.0])

    def test_bad_abundance_in_a_dataframe(self):
        """The same checks on the column types pandas produces: a nullable
        integer column with a gap is missing data, not text, and a column of
        numeric strings (a CSV read without parsing) is text."""
        gap = pd.DataFrame({"a": pd.array([1, pd.NA], dtype="Int64"), "b": [2, 3]})
        with pytest.raises(ValueError, match="missing or non-finite"):
            coleman(gap, [1.0, 2.0])
        with pytest.raises(ValueError, match="missing or non-finite"):
            coleman(pd.DataFrame({"a": [1.0, np.nan], "b": [2, 3]}), [1.0, 2.0])
        with pytest.raises(ValueError, match="numbers only"):
            coleman(pd.DataFrame({"a": ["1", "2"], "b": [2, 3]}), [1.0, 2.0])

    def test_nullable_integer_dataframe_without_gaps_is_accepted(self):
        counts, area = _simulate(3)
        df = pd.DataFrame(counts).astype("Int64")
        np.testing.assert_allclose(
            coleman(df, area).sites["expected"], coleman(counts, area).sites["expected"]
        )

    @pytest.mark.parametrize(
        ("area", "match"),
        [
            ([1.0], "one value per site"),
            ([1.0, 2.0, 3.0], "one value per site"),
            ([[1.0, 2.0]], "one value per site"),
            ([1.0, 0.0], "positive and finite"),
            ([1.0, -2.0], "positive and finite"),
            ([1.0, np.nan], "positive and finite"),
            ([1.0, np.inf], "positive and finite"),
        ],
    )
    def test_bad_area(self, area, match):
        with pytest.raises(ValueError, match=match):
            coleman([[2, 3], [1, 4]], area)

    def test_exported(self):
        assert sars.ColemanTest is ColemanTest
        assert sars.coleman is coleman
        assert {"coleman", "ColemanFit", "ColemanTest", "plot_coleman"} <= set(sars.__all__)
