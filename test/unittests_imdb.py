"""Unit Tests for IMDB Queries
To run the tests: go to the box-office root directory
run `make imdb`
"""
from __future__ import division
import unittest as unittest

class TestIMDB(unittest.TestCase):

    def test_perimeter(self):
        actual = 4.0
        answer = 4.0
        self.assertAlmostEqual(actual, answer)


if __name__ == '__main__':
    unittest.main()

# WHERE:
# tc = unittest.TestCase


# tc.addCleanup
# tc.assertGreater
# tc.assertLess
# tc.assertNotIsInstance
# tc.assert_
# tc.failUnless
# tc.setUp
# tc.addTypeEqualityFunc
# tc.assertGreaterEqual
# tc.assertLessEqual
# tc.assertNotRegexpMatches
# tc.countTestCases
# tc.failUnlessAlmostEqual
# tc.setUpClass
# tc.assertAlmostEqual
# tc.assertIn
# tc.assertListEqual
# tc.assertRaises
# tc.debug
# tc.failUnlessEqual
# tc.shortDescription
# tc.assertAlmostEquals
# tc.assertIs
# tc.assertMultiLineEqual
# tc.assertRaisesRegexp
# tc.defaultTestResult
# tc.failUnlessRaises
# tc.skipTest
# tc.assertDictContainsSubset
# tc.assertIsInstance
# tc.assertNotAlmostEqual
# tc.assertRegexpMatches
# tc.doCleanups
# tc.failureException
# tc.tearDown
# tc.assertDictEqual
# tc.assertIsNone
# tc.assertNotAlmostEquals
# tc.assertSequenceEqual
# tc.fail
# tc.id
# tc.tearDownClass
# tc.assertEqual
# tc.assertIsNot
# tc.assertNotEqual
# tc.assertSetEqual
# tc.failIf
# tc.longMessage
# tc.assertEquals
# tc.assertIsNotNone
# tc.assertNotEquals
# tc.assertTrue
# tc.failIfAlmostEqual
# tc.maxDiff
# tc.assertFalse
# tc.assertItemsEqual
# tc.assertNotIn
# tc.assertTupleEqual
# tc.failIfEqual
# tc.run
