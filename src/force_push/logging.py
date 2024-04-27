import pickle

import mobile_manipulation_central as mm


# fmt: off
ROSBAG_TOPICS = [
        "/clock",
        "--regex", "/ridgeback/(.*)",
        "--regex", "/ridgeback_velocity_controller/(.*)",
        "--regex", "/vicon/(.*)",
        "--regex", "/wrench/(.*)",
        "/time_msg",
]
# fmt: on


class DataRecorder(mm.DataRecorder):
    def __init__(self, name=None, notes=None, params=None):
        super().__init__(topics=ROSBAG_TOPICS, name=name, notes=notes)
        self.params = params

    def _record_params(self):
        if self.params is not None:
            filename = self.log_dir / "params.pkl"
            with open(filename, "wb") as f:
                pickle.dump(self.params, f)

    def record(self):
        super()._mkdir()
        super()._record_notes()
        self._record_params()
        super()._record_bag()

    def record_params(self, params):
        self.params = params
        self._record_params()
