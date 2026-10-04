"""Checks for the substantive corrections to the original workshop code."""

import unittest

import pandas as pd

from src.analysis import SHARE_FEATURES, TEXT_FEATURES, clickbait_score, make_features


class AnalysisTests(unittest.TestCase):
    def test_uppercase_headline_is_scored_before_lowercasing(self):
        self.assertEqual(clickbait_score("SHOCKING REVELATION!!!"), 3)
        self.assertEqual(clickbait_score("shocking revelation!!!"), 1)

    def test_regression_inputs_are_independent_of_sharing_target(self):
        frame = pd.DataFrame({"Title": ["An example title"], "Corps_Textee": ["Some body text"], "Nb_Partages": [10]})
        changed = frame.assign(Nb_Partages=10000)
        pd.testing.assert_frame_equal(make_features(frame)[TEXT_FEATURES], make_features(changed)[TEXT_FEATURES])
        self.assertFalse(set(TEXT_FEATURES) & set(SHARE_FEATURES))

    def test_share_clipping_uses_supplied_threshold(self):
        frame = pd.DataFrame({"Title": ["Title"], "Corps_Textee": ["abc"], "Nb_Partages": [10000]})
        features = make_features(frame, share_cap=100)
        self.assertEqual(features.iloc[0]["Shares_Per_Char"], 25)


if __name__ == "__main__":
    unittest.main()
