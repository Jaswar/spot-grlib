import numpy as np
from grlib.load_data.by_folder_loader import ByFolderLoader
from grlib.feature_extraction.pipeline import Pipeline
from grlib.filter.false_positive_filter import FalsePositiveFilter
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


def create_classifier(pipeline):
    # Create, train and test a model for static gesture recognition
    loader = ByFolderLoader(pipeline, path='./out')
    loader.create_landmarks()

    dataset = loader.load_landmarks()
    X = dataset.iloc[:, :63]
    y = dataset.iloc[:, 63]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    model = KNeighborsClassifier(5)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(preds, y_test)
    print(accuracy_score(preds, y_test))

    return model, dataset


# TODO: Maybe figure out how exactly it is going to stand before we run it (so we don't break something)
# TODO: The methods are synchronous now, maybe make them async somehow
def stand_low(command_client):
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.05)
    command_client.robot_command(cmd)


def stand_high(command_client):
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.2)
    command_client.robot_command(cmd)


def recognize_gestures(robot, model, pipeline, fp_filter):
    # Initialize an ImageClient
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()
    print(sources)

    # Create a command client to be able to command the robot
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    while True:
        # Retrieve image from the arm camera
        # TODO: Replace to match arm camera
        image_response = image_client.get_image_from_sources(['frontleft_fisheye_image'])
        image = np.array(Image.open(io.BytesIO(image_response.shot.image.data)))

        cv.imshow('Image', image)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Process the image through the pipeline and run prediction
        # TODO: maybe let get_world_landmarks_from_image already return flattened list so we dont have to do it here
        landmarks = pipeline.get_world_landmarks_from_image(image).flatten().tolist()
        pipeline.optimize()
        prediction = model.predict(landmarks)

        # Check if gesture is valid and if so do the corresponding action
        if fp_filter.is_relevant(landmarks):
            if prediction == model.classes_[0]:
                stand_low(command_client)
            else:
                stand_high(command_client)

    cv.destroyAllWindows()


def main():
    # Load the environment variables from .env
    load_dotenv()

    # Construct the model for gesture recognition
    pipeline = Pipeline(num_hands=1)
    pipeline.add_stage(0, 0)
    model, dataset = create_classifier(pipeline)

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

        # Create false-positive filter to recognize only meaningful gestures
        fp_filter = FalsePositiveFilter(dataset)
        # Start recognizing gestures
        recognize_gestures(robot, model, pipeline, fp_filter)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."


if __name__ == '__main__':
    main()
