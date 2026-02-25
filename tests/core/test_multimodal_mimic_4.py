from datetime import timedelta, datetime
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4

# I am creating a few mock classes because I feel like there is no current solution for 
# mocking up a patient. If this proves to be useful to other projects I would love
# for this to also live out of this unit test.

class MockNote:
    def __init__(self, text, timestamp):
        self.text = text
        self.timestamp = timestamp


class MockAdmission:
    def __init__(self, hadm_id, timestamp, hospital_expire_flag=0,
                 discharge_notes=None, radiology_notes=None):
        self.hadm_id = hadm_id
        self.timestamp = timestamp
        self.hospital_expire_flag = hospital_expire_flag
        self.discharge_notes = discharge_notes or []
        self.radiology_notes = radiology_notes or []


class MockPatient:
    def __init__(self, patient_id, admissions):
        self.patient_id = patient_id
        self._admissions = admissions
        self._demo = MagicMock()

    def get_events(self, event_type=None, filters=None, **kwargs):
        if event_type == "patients":
            return [self._demo]
        if event_type == "admissions":
            return self._admissions
        if filters:
            hadm_id = next((f[2] for f in filters if f[0] == "hadm_id"), None)
            admission = next((a for a in self._admissions if a.hadm_id == hadm_id), None)
            if admission:
                if event_type == "discharge":
                    return admission.discharge_notes
                if event_type == "radiology":
                    return admission.radiology_notes
        return []


_discharge_note = MockNote("Patient had pneumonia.", datetime(2023, 1, 2, 10, 0))  # 24h after hadm_1
_radiology_note = MockNote("CXR shows infiltrate.", datetime(2023, 1, 1, 14, 0))  # 4h after hadm_1

patient_no_notes = MockPatient("p1", admissions=[
    MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0)),
    MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),
])

patient_mixed_notes = MockPatient("p1", admissions=[
    MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                  discharge_notes=[_discharge_note],
                  radiology_notes=[_radiology_note]),
    MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),  # no notes
])

class TestClinicalNotesMIMIC4(unittest.TestCase):

    def test_missing_notes(self):
        """Both admissions have no notes — each gets <missing> and NaN placeholders."""
        import math

        samples = ClinicalNotesMIMIC4()(patient_no_notes)

        self.assertEqual(len(samples), 1)
        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertTrue(math.isnan(hadm_1_discharge_time))
        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertTrue(math.isnan(hadm_2_discharge_time))

        self.assertEqual(hadm_1_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertTrue(math.isnan(hadm_1_radiology_time))
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertTrue(math.isnan(hadm_2_radiology_time))

    def test_mixed_admission_notes(self):
        """First admission has notes, second does not — second gets <missing> and NaN placeholders."""
        import math

        samples = ClinicalNotesMIMIC4()(patient_mixed_notes)

        self.assertEqual(len(samples), 1)
        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge, "Patient had pneumonia.")
        self.assertAlmostEqual(hadm_1_discharge_time, 24.0)
        self.assertEqual(hadm_1_radiology, "CXR shows infiltrate.")
        self.assertAlmostEqual(hadm_1_radiology_time, 4.0)

        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertTrue(math.isnan(hadm_2_discharge_time))
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertTrue(math.isnan(hadm_2_radiology_time))

    def test_task_schema(self):
        self.assertIn("task_name", vars(ClinicalNotesMIMIC4))
        self.assertIn("input_schema", vars(ClinicalNotesMIMIC4))
        self.assertIn("output_schema", vars(ClinicalNotesMIMIC4))

        self.assertEqual("ClinicalNotesMIMIC4", ClinicalNotesMIMIC4.task_name)

        self.assertIn("discharge_note_times", ClinicalNotesMIMIC4.input_schema)
        self.assertIn("radiology_note_times", ClinicalNotesMIMIC4.input_schema)
        self.assertIn("mortality", ClinicalNotesMIMIC4.output_schema)
