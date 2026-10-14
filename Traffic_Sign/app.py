from flask import Flask, request, render_template, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained traffic sign detection model
try:
    logger.info("Loading model from traffic_sign_classifier.h5")
    model = load_model('traffic_sign_classifier.h5')
    logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Define class labels
class_labels = {
    0: 'ALL_MOTOR_VEHICLE_PROHIBITED',
    1: 'AXLE_LOAD_LIMIT',
    2: 'BARRIER_AHEAD',
    3: 'BULLOCK_AND_HANDCART_PROHIBITED',
    4: 'BULLOCK_PROHIBITED',
    5: 'CATTLE',
    6: 'COMPULSARY_AHEAD',
    7: 'COMPULSARY_AHEAD_OR_TURN_LEFT',
    8: 'COMPULSARY_AHEAD_OR_TURN_RIGHT',
    9: 'COMPULSARY_CYCLE_TRACK',
    10: 'COMPULSARY_KEEP_LEFT',
    11: 'COMPULSARY_KEEP_RIGHT',
    12: 'COMPULSARY_MINIMUM_SPEED',
    13: 'COMPULSARY_SOUND_HORN',
    14: 'COMPULSARY_TURN_LEFT',
    15: 'COMPULSARY_TURN_LEFT_AHEAD',
    16: 'COMPULSARY_TURN_RIGHT',
    17: 'COMPULSARY_TURN_RIGHT_AHEAD',
    18: 'CROSS_ROAD',
    19: 'CYCLE_CROSSING',
    20: 'CYCLE_PROHIBITED',
    21: 'DANGEROUS_DIP',
    22: 'DIRECTION',
    23: 'FALLING_ROCKS',
    24: 'FERRY',
    25: 'GAP_IN_MEDIAN',
    26: 'GIVE_WAY',
    27: 'GUARDED_LEVEL_CROSSING',
    28: 'HANDCART_PROHIBITED',
    29: 'HEIGHT_LIMIT',
    30: 'HORN_PROHIBITED',
    31: 'HUMP_OR_ROUGH_ROAD',
    32: 'LEFT_HAIR_PIN_BEND',
    33: 'LEFT_HAND_CURVE',
    34: 'LEFT_REVERSE_BEND',
    35: 'LEFT_TURN_PROHIBITED',
    36: 'LENGTH_LIMIT',
    37: 'LOAD_LIMIT',
    38: 'LOOSE_GRAVEL',
    39: 'MEN_AT_WORK',
    40: 'NARROW_BRIDGE',
    41: 'NARROW_ROAD_AHEAD',
    42: 'NO_ENTRY',
    43: 'NO_PARKING',
    44: 'NO_STOPPING_OR_STANDING',
    45: 'OVERTAKING_PROHIBITED',
    46: 'PASS_EITHER_SIDE',
    47: 'PEDESTRIAN_CROSSING',
    48: 'PEDESTRIAN_PROHIBITED',
    49: 'PRIORITY_FOR_ONCOMING_VEHICLES',
    50: 'QUAY_SIDE_OR_RIVER_BANK',
    51: 'RESTRICTION_ENDS',
    52: 'RIGHT_HAIR_PIN_BEND',
    53: 'RIGHT_HAND_CURVE',
    54: 'RIGHT_REVERSE_BEND',
    55: 'RIGHT_TURN_PROHIBITED',
    56: 'ROAD_WIDENS_AHEAD',
    57: 'ROUNDABOUT',
    58: 'SCHOOL_AHEAD',
    59: 'SIDE_ROAD_LEFT',
    60: 'SIDE_ROAD_RIGHT',
    61: 'SLIPPERY_ROAD',
    62: 'SPEED_LIMIT_15',
    63: 'SPEED_LIMIT_20',
    64: 'SPEED_LIMIT_30',
    65: 'SPEED_LIMIT_40',
    66: 'SPEED_LIMIT_5',
    67: 'SPEED_LIMIT_50',
    68: 'SPEED_LIMIT_60',
    69: 'SPEED_LIMIT_70',
    70: 'SPEED_LIMIT_80',
    71: 'STAGGERED_INTERSECTION',
    72: 'STEEP_ASCENT',
    73: 'STEEP_DESCENT',
    74: 'STOP',
    75: 'STRAIGHT_PROHIBITED',
    76: 'TONGA_PROHIBITED',
    77: 'TRAFFIC_SIGNAL',
    78: 'TRUCK_PROHIBITED',
    79: 'TURN_RIGHT',
    80: 'T_INTERSECTION',
    81: 'UNGUARDED_LEVEL_CROSSING',
    82: 'U_TURN_PROHIBITED',
    83: 'WIDTH_LIMIT',
    84: 'Y_INTERSECTION'
}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_traffic_sign(img_path):
    try:
        logger.info(f"Processing image: {img_path}")

        # Get input shape from model
        input_shape = model.input_shape[1:3]
        logger.info(f"Model input shape: {input_shape}")

        # Load and preprocess the image
        img = image.load_img(img_path, target_size=input_shape)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Make prediction
        logger.info("Running prediction")
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

        logger.info(f"Predicted class: {predicted_class}, Confidence: {confidence}")

        # Confidence threshold
        if confidence < 0.8:
            logger.info("Confidence too low, no sign detected")
            return {
                "success": False,
                "message": "No traffic sign detected in the uploaded image or confidence too low."
            }
        else:
            logger.info(f"Sign detected: {class_labels.get(predicted_class)}")
            return {
                "success": True,
                "class": int(predicted_class),
                "label": class_labels.get(predicted_class, 'Unknown Sign'),
                "confidence": float(confidence)
            }

    except Exception as e:
        logger.error(f"Error processing the image: {str(e)}")
        return {
            "success": False,
            "message": f"Error processing the image: {str(e)}"
        }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    logger.info("Received prediction request")

    if 'file' not in request.files:
        logger.warning("No file part in request")
        return jsonify({'success': False, 'message': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'success': False, 'message': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to {filepath}")
        file.save(filepath)

        # Make prediction
        result = predict_traffic_sign(filepath)

        # Optional: Remove the file after prediction
        # os.remove(filepath)

        return jsonify(result)

    logger.warning("File type not allowed")
    return jsonify({'success': False, 'message': 'File type not allowed'})


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)