# Description: This script is used to create a dataset for the project.

# =================================================================================================

from CamSupport.tinder import TinderSession

# =================================================================================================

if __name__ == "__main__":
    session = TinderSession(out_path="Data/Test", classes=("IO", "NIO"), tag="A01")
    session.start_gui()