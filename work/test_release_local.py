"""Exercise code-only deployment rollback without invoking Docker or HA."""
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from release_local import backup, local_mapping, pending, restore_code


class ReleaseLocalTests(unittest.TestCase):
    def test_rollback_preserves_commands_and_database_written_after_backup(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        shadow, pilot, data = map(lambda name: root/name, ("shadow", "pilot-data", "shadow-data"))
        for directory in (shadow, pilot, data):
            directory.mkdir()
        (shadow/"continuous_pilot.py").write_text("old = True\n")
        (shadow/"rooms.json").write_text("{}\n")
        journal = pilot/"continuous-1.json"
        journal.write_text('{"commands": []}')
        db = sqlite3.connect(data/"shadow.sqlite3")
        self.addCleanup(db.close)
        db.execute("CREATE TABLE observations (id INTEGER)")
        db.commit()
        target = root/"backup"
        backup(root, target, {"continuous_pilot.py"})
        (shadow/"continuous_pilot.py").write_text("old = False\n")
        (shadow/".release").write_text("new")
        journal.write_text('{"commands": [{"room": "living", "sent_at": 2, "status": "accepted"}]}')
        db.execute("INSERT INTO observations VALUES (1)")
        db.commit()
        self.assertEqual(pending(root), [("living", 2, "accepted")])
        restore_code(shadow, target/"runtime")
        self.assertEqual((shadow/"continuous_pilot.py").read_text(), "old = True\n")
        self.assertFalse((shadow/".release").exists())
        self.assertEqual(pending(root), [("living", 2, "accepted")])
        self.assertEqual(db.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 1)

    def test_migrate_legacy_bindings_preserves_existing_private_values(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        (root/"rooms.json").write_text('{"living":{"power":"sensor.local"}}')
        (root/"pilot_gate.py").write_text(
            'LIVING_REFERENCE="sensor.reference"\nTEMPERATURE={"living":LIVING_REFERENCE}\n'
            'WITNESSES={"living":"fixture"}\n')
        self.assertEqual(json.loads(local_mapping(root)), {"living":{
            "power":"sensor.local", "temperature":"sensor.reference", "witness_trial_id":"fixture"}})


if __name__ == "__main__":
    unittest.main()
