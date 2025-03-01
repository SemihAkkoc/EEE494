from AIConfirmation import *
from nonAIScan import *
from utils import *
import threading
from time import sleep, time

Lock = threading.Lock()
undeterminedDroneCandidateList = []
droneCandidateList = []
startCode = False

# This function will run the AIConfirmation thread
def AIScanThreadFunction(undeterminedDroneCandidateList : list):
    last_motor_angles = (0, 0)
    while True:
        if len(undeterminedDroneCandidateList) == 0:
            sleep(1) # if there is no isDrone=None drone candidate then sleep for 1 second
        else:
            # else, get the first element of the undeterminedDroneCandidateList and run the AIConfirmation function
            droneCandidate = undeterminedDroneCandidateList[0]
            # Confirm the candidate with AI camera, update the object
            last_motor_angles = AIConfirmation(droneCandidate, last_motor_angles) # updates the isDrone flag
            with Lock:
                # remove the candidate from the undeterminedDroneCandidateList
                undeterminedDroneCandidateList.pop(0)


def main():
    # initialise the buttons
    initialiseAllSystems()
    startStopThread = threading.Thread(target=isButtonPressed) # This button changes the state of the startCode
    startStopThread.start()

    # wait for the start code
    while not startCode:
        sleep(0.1)

    # start the AIConfirmation thread
    AIConfirmationThread = threading.Thread(target=AIScanThreadFunction, args=(undeterminedDroneCandidateList,))
    AIConfirmationThread.start()

    # start showing the results via LED screen
    LEDScreenShowThread = threading.Thread(target=LEDScreenShow, args=(droneCandidateList,))
    LEDScreenShowThread.start()

    # start the non-AI scan with while loop
    nonAIScan()

    # add stop protocols
    startStopThread.join()
    AIConfirmationThread.join()
    LEDScreenShowThread.join()

if __name__ == "__`main__":
    main()
