from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import nodes


class RuntimeLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop(nodes.RUNTIME_MODULE_NAME, None)

    def tearDown(self) -> None:
        sys.modules.pop(nodes.RUNTIME_MODULE_NAME, None)

    def _runtime_root(
        self, source_text: str
    ) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        runtime_root = Path(temporary.name)
        scripts = runtime_root / "scripts"
        scripts.mkdir()
        (scripts / "generalized-full-image-refine.py").write_text(
            source_text,
            encoding="utf-8",
        )
        return temporary, runtime_root

    def test_reuses_only_a_complete_module_from_the_expected_source(self) -> None:
        temporary, runtime_root = self._runtime_root(
            "def refine_array(*args, **kwargs):\n    return args, kwargs\n"
        )
        self.addCleanup(temporary.cleanup)
        source = runtime_root / "scripts" / "generalized-full-image-refine.py"
        cached = types.ModuleType(nodes.RUNTIME_MODULE_NAME)
        cached.__file__ = str(source)
        cached.refine_array = lambda: "cached"
        sys.modules[nodes.RUNTIME_MODULE_NAME] = cached

        with mock.patch.object(nodes, "_runtime_root", return_value=runtime_root):
            self.assertIs(nodes._load_refiner(), cached)

    def test_reloads_a_partially_initialized_cached_module(self) -> None:
        temporary, runtime_root = self._runtime_root(
            "def refine_array(*args, **kwargs):\n    return 'loaded'\n"
        )
        self.addCleanup(temporary.cleanup)
        stale = types.ModuleType(nodes.RUNTIME_MODULE_NAME)
        stale.__file__ = str(
            runtime_root / "scripts" / "generalized-full-image-refine.py"
        )
        sys.modules[nodes.RUNTIME_MODULE_NAME] = stale

        with mock.patch.object(nodes, "_runtime_root", return_value=runtime_root):
            loaded = nodes._load_refiner()

        self.assertIsNot(loaded, stale)
        self.assertTrue(callable(loaded.refine_array))
        self.assertEqual(loaded.refine_array(), "loaded")

    def test_failed_import_removes_the_partial_module_and_reports_versions(
        self,
    ) -> None:
        temporary, runtime_root = self._runtime_root(
            "raise ImportError('numpy.core.multiarray failed to import')\n"
        )
        self.addCleanup(temporary.cleanup)

        with (
            mock.patch.object(nodes, "_runtime_root", return_value=runtime_root),
            self.assertRaisesRegex(
                RuntimeError,
                r"numpy=.*OpenCV distributions:.*numpy\.core\.multiarray failed to import",
            ),
        ):
            nodes._load_refiner()

        self.assertNotIn(nodes.RUNTIME_MODULE_NAME, sys.modules)


if __name__ == "__main__":
    unittest.main()
