from datetime import timedelta, datetime
from pathlib import Path
from unittest.mock import MagicMock
import tempfile
import unittest

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4, ClinicalNotesICDLabsMIMIC4

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

class TestClinicalNotesICDLabsMIMIC4(unittest.TestCase):
    pass

class TestClinicalNotesMIMIC4(unittest.TestCase):

    mock_discharge_note = MockNote("Patient had pneumonia.", datetime(2023, 1, 2, 10, 0))   # 24h after hadm_1
    mock_discharge_note_2 = MockNote("Follow-up: resolving pneumonia.", datetime(2023, 1, 3, 10, 0))  # 48h after hadm_1
    mock_radiology_note = MockNote("CXR shows infiltrate.", datetime(2023, 1, 1, 14, 0))   # 4h after hadm_1

    patient_no_notes = MockPatient("p1", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0)),
        MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),
    ])

    patient_with_notes_in_first_admisssion_but_not_the_second = MockPatient("p1", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                    discharge_notes=[mock_discharge_note],
                    radiology_notes=[mock_radiology_note]),
        MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),  # no notes
    ])

    patient_with_multiple_discharge_notes_in_first_admission = MockPatient("p1", admissions=[
        MockAdmission("hadm_1", datetime(2023, 1, 1, 10, 0),
                    discharge_notes=[mock_discharge_note, mock_discharge_note_2],
                    radiology_notes=[mock_radiology_note]),
        MockAdmission("hadm_2", datetime(2023, 2, 1, 10, 0)),  # no notes
    ])

    def test_both_admission_missing_notes(self):
        """When a patient has two admissions with no notes for either,
        each admission should produce the missing text and float placeholders
        defined by TOKEN_REPRESENTING_MISSING_TEXT and TOKEN_REPRESENTING_MISSING_FLOAT."""
        samples = ClinicalNotesMIMIC4()(self.patient_no_notes)

        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_1_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

        self.assertEqual(hadm_1_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_1_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

    def test_first_admission_has_notes_second_does_not(self):
        """When a patient has two admissions and only the first has notes,
        the first admission should produce real note text and time offsets in hours,
        while the second should produce the missing text and float placeholders."""
        samples = ClinicalNotesMIMIC4()(self.patient_with_notes_in_first_admisssion_but_not_the_second)

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
        self.assertEqual(hadm_2_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

    def test_multiple_discharge_notes_per_admission(self):
        """hadm_1 has 2 discharge notes (at 24h and 48h) and 1 radiology note (at 4h).
        hadm_2 has no notes. Verifies that both discharge notes from hadm_1 are collected
        in order with correct time offsets, and that hadm_2 gets the missing text and float
        tokens for both discharge and radiology."""
        samples = ClinicalNotesMIMIC4()(self.patient_with_multiple_discharge_notes_in_first_admission)

        discharge_texts, discharge_times = samples[0]["discharge_note_times"]
        radiology_texts, radiology_times = samples[0]["radiology_note_times"]

        hadm_1_discharge_1, hadm_1_discharge_2, hadm_2_discharge = discharge_texts
        hadm_1_discharge_time_1, hadm_1_discharge_time_2, hadm_2_discharge_time = discharge_times
        hadm_1_radiology, hadm_2_radiology = radiology_texts
        hadm_1_radiology_time, hadm_2_radiology_time = radiology_times

        self.assertEqual(hadm_1_discharge_1, "Patient had pneumonia.")
        self.assertAlmostEqual(hadm_1_discharge_time_1, 24.0)
        self.assertEqual(hadm_1_discharge_2, "Follow-up: resolving pneumonia.")
        self.assertAlmostEqual(hadm_1_discharge_time_2, 48.0)

        self.assertEqual(hadm_2_discharge, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_discharge_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

        self.assertEqual(hadm_1_radiology, "CXR shows infiltrate.")
        self.assertAlmostEqual(hadm_1_radiology_time, 4.0)
        self.assertEqual(hadm_2_radiology, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_TEXT)
        self.assertEqual(hadm_2_radiology_time, ClinicalNotesMIMIC4.TOKEN_REPRESENTING_MISSING_FLOAT)

    def test_task_schema(self):
        """Verify that ClinicalNotesMIMIC4 exposes the expected class-level schema attributes
        with the correct keys for inputs (discharge and radiology note times) and output (mortality)."""
        self.assertIn("task_name", vars(ClinicalNotesMIMIC4))
        self.assertIn("input_schema", vars(ClinicalNotesMIMIC4))
        self.assertIn("output_schema", vars(ClinicalNotesMIMIC4))

        self.assertEqual("ClinicalNotesMIMIC4", ClinicalNotesMIMIC4.task_name)

        self.assertIn("discharge_note_times", ClinicalNotesMIMIC4.input_schema)
        self.assertIn("radiology_note_times", ClinicalNotesMIMIC4.input_schema)
        self.assertIn("mortality", ClinicalNotesMIMIC4.output_schema)
