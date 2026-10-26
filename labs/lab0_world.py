"""
LAB 0: THE WORLD. Who holds what, and why pooling is off the table.

    python labs/lab0_world.py

Prints the three hospitals, what each one holds, one patient in human units
(to make it real), and the coordinator's neutral test set. Nothing is trained.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl import ui
from fl.world import make_world, to_human

world = make_world(3, "realistic")

ui.title("LAB 0  |  The readmission consortium")
print(world.describe())

ui.section("What a hospital holds: 24 clinical features per discharge, already standardized")
print(", ".join(world.features))

ui.section("One patient at Kestrel, in human units (synthetic, nobody real)")
for k, v in to_human(world.clients[2].X_train[0]).items():
    print(f"  {k:<24} {v}")
print(f"  {'readmitted_30d':<24} {'yes' if world.clients[2].y_train[0] == 1 else 'no'}")

ui.section("What the coordinator holds")
print("  A neutral, consented evaluation set: 500 de-identified rows from each hospital's population.")
print("  Zero training records. The coordinator never sees a hospital's patients.")

ui.notice([
    "The hospitals differ before any learning happens: size, readmission rate, population.",
    "Kestrel has 60 patients. Alone, that is not enough to learn 24 weights well.",
    "Northgate has 1,800. Why would Northgate bother joining? Hold that question for Lab 1.",
])
ui.note_prompt(0, "In one sentence: what does the coordinator hold, and what does it never hold?")
