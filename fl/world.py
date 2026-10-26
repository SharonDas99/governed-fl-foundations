"""
THE WORLD: a readmission consortium that cannot pool its data.

Several hospitals want one model that predicts 30-day readmission after
discharge. Each hospital keeps its own patient records. A coordinator holds
a small NEUTRAL test set (de-identified, consented, for evaluation only).

Everything is synthetic. Nobody here is a real patient.

What you get from make_world(...)
  world.clients        list of Client objects, one per hospital:
                         .name, .X_train, .y_train   (that is ALL a hospital has)
  world.X_test, .y_test  the coordinator's neutral test set
  world.test_source    which hospital's population each test row came from, so
                       the coordinator can report accuracy PER hospital
  world.features       the 24 clinical feature names (what hospitals hold)
  world.claims_features  6 insurer features (Lab 4, vertical FL only)
  world.X_claims_*     the insurer's view of the SAME patients (Lab 4)

Features are already standardized (z-scores against a public reference
population), which is what a real consortium would agree on in advance so
that nobody has to share raw means. to_human() converts back for display.

Skew modes (Lab 3). They change how DIFFERENT the hospitals are:
  none       identical distributions, equal sizes          (the textbook IID case)
  label      same patients, different readmission rates    (label skew)
  quantity   same distribution, very different sizes       (quantity skew)
  feature    one hospital's patients are older and sicker  (covariate shift)
  concept    one hospital where a feature means the opposite (concept shift)
  realistic  a mix of all of the above, mild (the default everywhere else)
"""
from dataclasses import dataclass

import numpy as np

SEED = 690

HOSPITALS = ["Northgate", "Three Rivers", "Kestrel", "Pine Valley", "Harbor General",
             "Mesa County", "Lakeside", "Summit Ridge", "Cedar Point", "Westbrook"]

CLINICAL = ["age", "prior_admissions_12m", "length_of_stay_days", "num_medications",
            "comorbidity_index", "er_visits_6m", "hemoglobin", "sodium", "creatinine",
            "bmi", "lives_alone", "discharged_to_facility",
            "systolic_bp", "heart_rate", "respiratory_rate", "temperature", "white_cell_count",
            "glucose", "potassium", "albumin", "oxygen_saturation", "pain_score",
            "mobility_score", "followup_scheduled"]
CLAIMS = ["claims_12m", "total_paid_12m", "pharmacy_fills_90d", "outpatient_visits_12m",
          "days_since_last_claim", "plan_hmo"]

# True effect of each standardized feature on the readmission log-odds.
BETA_CLINICAL = np.array([0.30, 0.90, 0.40, 0.35, 0.70, 0.60, -0.30, -0.20, 0.30, 0.10, 0.30, 0.25,
                          0.10, 0.15, 0.12, 0.05, 0.12, 0.10, -0.08, -0.25, -0.15, 0.10, -0.20, -0.30])
BETA_CLAIMS = np.array([0.50, 0.30, 0.20, -0.10, -0.30, 0.00])
BASE_INTERCEPT = -1.05          # about 30 percent readmission in the reference population
NOISE_SD = 1.3

# Human units for display only: (mean, sd) of each standardized feature.
HUMAN = {
    "age": (67, 14), "prior_admissions_12m": (1.1, 1.3), "length_of_stay_days": (4.8, 3.1),
    "num_medications": (9, 4), "comorbidity_index": (2.4, 1.8), "er_visits_6m": (0.9, 1.2),
    "hemoglobin": (12.4, 1.8), "sodium": (138, 3.5), "creatinine": (1.2, 0.6), "bmi": (28.5, 6.0),
    "lives_alone": (0.3, 0.46), "discharged_to_facility": (0.2, 0.4),
    "systolic_bp": (132, 20), "heart_rate": (82, 14), "respiratory_rate": (18, 3), "temperature": (37.0, 0.5),
    "white_cell_count": (8.5, 3.0), "glucose": (128, 45), "potassium": (4.1, 0.5), "albumin": (3.6, 0.5),
    "oxygen_saturation": (95, 3), "pain_score": (3, 2.5), "mobility_score": (6, 2.5), "followup_scheduled": (0.6, 0.49),
    "claims_12m": (6.2, 4.0), "total_paid_12m": (14800, 11000), "pharmacy_fills_90d": (7.5, 5.0),
    "outpatient_visits_12m": (5.1, 3.4), "days_since_last_claim": (48, 40), "plan_hmo": (0.45, 0.5),
}


@dataclass
class Client:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray

    @property
    def n(self):
        return len(self.y_train)

    def describe(self):
        return f"{self.name:<15} patients n={self.n:<5} readmission rate {self.y_train.mean():.2f}"


@dataclass
class World:
    clients: list
    X_test: np.ndarray
    y_test: np.ndarray
    test_source: np.ndarray       # index of the hospital each test row came from
    features: list
    claims_features: list
    X_claims_train: dict          # Lab 4: insurer's view of each hospital's TRAIN patients
    X_claims_test: np.ndarray     # Lab 4: insurer's view of the neutral test set
    skew: str

    def test_for(self, k):
        """The slice of the neutral test set that came from hospital k."""
        m = self.test_source == k
        return self.X_test[m], self.y_test[m]

    def describe(self):
        lines = [f"World: {len(self.clients)} hospitals, skew = {self.skew}"]
        lines += ["  " + c.describe() for c in self.clients]
        lines.append(f"  {'neutral test':<15} n={len(self.y_test):<5} readmission rate {self.y_test.mean():.2f}"
                     f"   ({len(self.y_test) // len(self.clients)} rows from each hospital's population)")
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Hospital profiles: how each hospital's population differs under each skew.
# ----------------------------------------------------------------------------
def _profiles(n_clients, skew, rng):
    """Returns per-hospital (size, intercept_shift, feature_shift vector, concept_flip)."""
    sizes = np.full(n_clients, 1000)
    intercepts = np.zeros(n_clients)
    shifts = np.zeros((n_clients, len(CLINICAL)))
    concept = np.zeros(n_clients, dtype=bool)

    def kestrel_shift():
        s = np.zeros(len(CLINICAL))
        s[CLINICAL.index("age")] = 0.8
        s[CLINICAL.index("comorbidity_index")] = 0.6
        s[CLINICAL.index("lives_alone")] = 0.5
        s[CLINICAL.index("hemoglobin")] = -0.4
        return s

    if skew == "label":
        intercepts[:3] = [-0.9, 0.0, 0.9]
    elif skew == "quantity":
        sizes[:3] = [2400, 600, 60]
    elif skew == "feature":
        shifts[2] = kestrel_shift()
    elif skew == "concept":
        concept[2] = True
    elif skew == "realistic":
        sizes[:3] = [1800, 700, 60]
        intercepts[:3] = [-0.3, 0.1, 0.2]
        shifts[2] = 0.15 * kestrel_shift()
    # extra hospitals beyond the named three: mild random differences
    for k in range(3, n_clients):
        sizes[k] = int(rng.integers(300, 1500))
        if skew in ("label", "realistic"):
            intercepts[k] = rng.normal(0, 0.4)
        if skew in ("feature", "realistic"):
            shifts[k] = rng.normal(0, 0.25, len(CLINICAL))
        if skew == "quantity":
            sizes[k] = int(rng.integers(80, 2500))
    return sizes, intercepts, shifts, concept


def _draw(n, intercept, shift, concept_flip, rng):
    """Draw n patients: standardized clinical + claims features and the label."""
    Xc = rng.normal(0, 1, (n, len(CLINICAL))) + shift
    # binary-ish fields stay roughly binary in z-space
    Xk = rng.normal(0, 1, (n, len(CLAIMS)))
    # claims correlate with clinical burden (same patients, different lens)
    Xk[:, 0] += 0.5 * Xc[:, CLINICAL.index("prior_admissions_12m")]
    Xk[:, 1] += 0.4 * Xc[:, CLINICAL.index("length_of_stay_days")]
    Xk[:, 2] += 0.5 * Xc[:, CLINICAL.index("num_medications")]
    beta_c = BETA_CLINICAL.copy()
    if concept_flip:
        beta_c[CLINICAL.index("discharged_to_facility")] *= -1   # in this hospital, a facility discharge means good follow-up
        beta_c[CLINICAL.index("lives_alone")] *= -1
    logit = BASE_INTERCEPT + intercept + Xc @ beta_c + Xk @ BETA_CLAIMS + rng.normal(0, NOISE_SD, n)
    y = (logit > 0).astype(np.int64)
    return Xc.astype(np.float32), Xk.astype(np.float32), y


def make_world(n_clients=3, skew="realistic", seed=SEED, test_per_hospital=500):
    assert 2 <= n_clients <= len(HOSPITALS), f"n_clients must be 2..{len(HOSPITALS)}"
    rng = np.random.default_rng(seed)
    sizes, intercepts, shifts, concept = _profiles(n_clients, skew, rng)

    clients, claims_train = [], {}
    for k in range(n_clients):
        Xc, Xk, y = _draw(int(sizes[k]), intercepts[k], shifts[k], concept[k], rng)
        clients.append(Client(HOSPITALS[k], Xc, y))
        claims_train[HOSPITALS[k]] = Xk

    # Neutral test set: an equal number of rows from every hospital's population.
    Xt, Kt, yt, src = [], [], [], []
    for k in range(n_clients):
        Xc, Xk, y = _draw(test_per_hospital, intercepts[k], shifts[k], concept[k], rng)
        Xt.append(Xc); Kt.append(Xk); yt.append(y); src.append(np.full(test_per_hospital, k))
    return World(clients, np.vstack(Xt), np.concatenate(yt), np.concatenate(src), list(CLINICAL), list(CLAIMS),
                 claims_train, np.vstack(Kt), skew)


def to_human(x, names=CLINICAL):
    """Convert a standardized feature vector to approximate human units for display."""
    out = {}
    for v, nm in zip(x, names):
        m, s = HUMAN[nm]
        val = m + s * float(v)
        if nm in ("lives_alone", "discharged_to_facility", "plan_hmo", "followup_scheduled"):
            out[nm] = "yes" if val > 0.5 else "no"
        elif nm in ("prior_admissions_12m", "num_medications", "er_visits_6m", "claims_12m",
                    "pharmacy_fills_90d", "outpatient_visits_12m", "days_since_last_claim", "age",
                    "systolic_bp", "heart_rate", "respiratory_rate", "glucose", "oxygen_saturation",
                    "pain_score", "mobility_score"):
            out[nm] = int(round(max(val, 0)))
        elif nm == "total_paid_12m":
            out[nm] = int(round(max(val, 0) / 100) * 100)
        else:
            out[nm] = round(val, 1)
    return out


def pooled(world):
    """The forbidden ideal: every hospital's training data in one table."""
    X = np.vstack([c.X_train for c in world.clients])
    y = np.concatenate([c.y_train for c in world.clients])
    return X, y


if __name__ == "__main__":
    for skew in ["none", "label", "quantity", "feature", "concept", "realistic"]:
        print(make_world(3, skew).describe())
        print()
