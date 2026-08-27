"""Proof that the table protocol is full stay, with no observation-window API.

Collection is admit through discharge. Concatenated stays still share one
clock (see test_p1_time_axis.py). CXR arms do not skip later stays.

Repro::

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \\
      python -m pytest tests/test_p1_observation_window.py -q
"""

from __future__ import annotations

import inspect
import json
import unittest
import uuid
from datetime import datetime, timedelta


class TestP1FullStayProtocol(unittest.TestCase):
    def test_tasks_have_no_window_hours(self):
        from pyhealth.tasks.multimodal_mimic4 import (
            CXRMIMIC4,
            LabsMIMIC4,
            NotesLabsCXRMIMIC4,
            NotesLabsMIMIC4,
        )

        for cls in (NotesLabsMIMIC4, NotesLabsCXRMIMIC4, LabsMIMIC4, CXRMIMIC4):
            task = cls()
            self.assertFalse(hasattr(task, "window_hours"))
            self.assertNotIn("window_hours", inspect.signature(cls).parameters)

    def test_collection_end_is_discharge(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        task = LabsMIMIC4()
        admit = datetime(2180, 5, 6, 8, 0, 0)
        discharge = admit + timedelta(days=9)
        self.assertEqual(task._admission_window_end(admit, discharge), discharge)

    def test_cxr_arms_do_not_skip_later_stays(self):
        from pyhealth.tasks.multimodal_mimic4 import CXRMIMIC4, NotesLabsCXRMIMIC4

        for cls in (NotesLabsCXRMIMIC4, CXRMIMIC4):
            src = inspect.getsource(cls.__call__)
            self.assertNotIn("admission_time >= effective_end", src)

    def test_cache_version_invalidates_windowed_caches(self):
        from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4

        task = LabsMIMIC4()
        self.assertGreaterEqual(task.emitted_data_version, 5)

        def cache_key(t, drop_version=False):
            v = dict(vars(t))
            if drop_version:
                v.pop("emitted_data_version", None)
            params = json.dumps(
                {
                    **v,
                    "input_schema": t.input_schema,
                    "output_schema": t.output_schema,
                },
                sort_keys=True,
                default=str,
            )
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, params))

        self.assertNotEqual(cache_key(task), cache_key(task, drop_version=True))

    def test_discharge_coded_icd_is_not_a_mortality_task(self):
        from pyhealth.tasks import multimodal_mimic4 as m
        from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4

        self.assertFalse(hasattr(m, "ICDLabsMIMIC4"))
        self.assertFalse(NotesLabsMIMIC4().include_icd)
