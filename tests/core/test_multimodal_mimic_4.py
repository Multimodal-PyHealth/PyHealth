from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4

class TestClinicalNotesMIMIC4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory()

        dataset = MIMIC4Dataset(
            ehr_root=str(Path(__file__).parent.parent.parent / "test-resources" / "core" / "mimic4demo"),
            ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            # note_root="/path/to/mimic-iv-note/2.2",
            # note_tables=["discharge", "radiology"]

            cache_dir=cls.cache_dir.name,
        )
    
    def test_task_schema(self):
        task = ClinicalNotesMIMIC4()

        self.assertTrue(hasattr(task, "task_name"))
        self.assertTrue(hasattr(task, "input_schema"))
        self.assertTrue(hasattr(task, "output_schema"))

        self.assertEqual("ClinicalNotesMIMIC4", task.task_name)
        self.assertIn("discharge_note_times", task.input_schema)
        self.assertIn("radiology_note_times", task.input_schema)

