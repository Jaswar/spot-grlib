import numpy as np
from grlib.exceptions import NoHandDetectedException
from grlib.load_data.by_folder_loader import ByFolderLoader
from grlib.feature_extraction.pipeline import Pipeline
from grlib.filter.false_positive_filter import FalsePositiveFilter
from grlib.dynamic_detection import DynamicDetector
from grlib.load_data.dynamic_gesture_loader import DynamicGestureLoader
from grlib.trajectory.trajectory_classifier import TrajectoryClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from dotenv import load_dotenv
import os
import io
from PIL import Image
import cv2.cv2 as cv


ZERO_PRECISION = 0.1
NUM_HANDS = 1

# TODO: Maybe figure out how exactly it is going to stand before we run it (so we don't break something)
# TODO: The methods are synchronous now, maybe make them async somehow
def stand_low(command_client):
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.0)
    command_client.robot_command(cmd)


def stand_high(command_client):
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.5)
    command_client.robot_command(cmd)


def move_right(command_client):
    # velocity_command is deprecated so we will see what happens
    cmd = RobotCommandBuilder.velocity_command(5, 0, 0)
    command_client.robot_command_async(cmd, 10)


def move_left(command_client):
    # velocity_command is deprecated so we will see what happens
    cmd = RobotCommandBuilder.velocity_command(-5, 0, 0)
    command_client.robot_command_async(cmd, 10)


def stop(command_client):
    cmd = RobotCommandBuilder.stop_command()
    command_client.robot_command(cmd)


def recognize_gestures(robot, pipeline):
    loader = DynamicGestureLoader(
        pipeline, 'left_right_dataset', trajectory_zero_precision=ZERO_PRECISION, key_frames=3
    )
    #loader.create_landmarks()

    landmarks = loader.load_landmarks()
    trajectories = loader.load_trajectories()
    x_traj = trajectories.iloc[:, :-1]
    y = np.array(trajectories['label'])

    trajectory_classifier = TrajectoryClassifier()
    trajectory_classifier.fit(np.array(x_traj), y)

    start_shapes = loader.get_start_shape(landmarks, NUM_HANDS)

    detector = DynamicDetector(
        start_shapes,
        y,
        pipeline,
        start_pos_confidence=0.1,
        trajectory_classifier=trajectory_classifier,
        update_candidates_every=5,
        candidate_zero_precision=ZERO_PRECISION,
    )

    # Initialize an ImageClient
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()

    # Create a command client to be able to command the robot
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    while True:
        # Retrieve image from the arm camera
        image_response = image_client.get_image_from_sources(['hand_color_image'])[0]
        image = np.array(Image.open(io.BytesIO(image_response.shot.image.data)))[:, :, ::-1]

        cv.imshow('Image', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Process the image through the pipeline and run prediction
        try:
            detector.analyze_frame(image)
            prediction = detector.last_pred
            if prediction == 'left':
                move_left(command_client)
            elif prediction == 'right':
                move_right(command_client)
            else:
                stop(command_client)

        except NoHandDetectedException:
            print('No hand')

    cv.destroyAllWindows()


def main():
    # Load the environment variables from .env
    load_dotenv()

    # Construct the model for gesture recognition
    pipeline = Pipeline(num_hands=1)
    pipeline.add_stage(0, 0)

    # Set up the boston dynamics SPOT (based on hello_spot.py)
    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot(os.getenv('hostname'))
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Power on and start controlling the robot
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."

        # Start recognizing gestures
        recognize_gestures(robot, pipeline)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."


if __name__ == '__main__':
    main()
