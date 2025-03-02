import cv2
import numpy as np
import face_recognition
import dlib
import cx_Oracle
from flask import Flask, render_template, Response, request, jsonify
import threading
import time
from ultralytics import YOLO
from twilio.rest import Client

# Load YOLO model for people detection
model = YOLO("yolov8n.pt")

# Connect to Oracle SQL*Plus database
def get_db_connection():
    dsn_tns = cx_Oracle.makedsn('localhost', '1521', service_name='XE')  # Modify as needed
    return cx_Oracle.connect(user='system', password='thanuj', dsn=dsn_tns)
# Flask app
app = Flask(__name__)

guest_verification_enabled = False
monitoring_area = ((0, 0), (640, 480))  # Default perimeter
crowd_limit = 10  # Default crowd limit

# Twilio setup
TWILIO_SID = "enter ssid of your twilio account"
TWILIO_AUTH_TOKEN = "enter auth token "
TWILIO_PHONE = "enter your twilio phone number"
ALERT_PHONE = "enter admin phone number with country code at beginning like +91....."

def send_alert(message):
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    client.messages.create(body=message, from_=TWILIO_PHONE, to=ALERT_PHONE)

# Process video stream
current_people_count = 0  # Global variable to store live count

def process_frame(frame):
    global monitoring_area, crowd_limit, current_people_count
    results = model(frame)  # Run YOLO detection
    people_count = 0

    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])  
            label = r.names[class_id]
            
            if label == 'person':  # Ensure it's a person
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates

                # Ensure detection is within the monitoring area
                if monitoring_area[0][0] <= x1 <= monitoring_area[1][0] and monitoring_area[0][1] <= y1 <= monitoring_area[1][1]:
                    people_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Update global count after processing all detections
    current_people_count = people_count  

    # Check if an alert needs to be sent
    if people_count > crowd_limit:
        send_crowd_alert(f"Crowd alert! {people_count} people detected.")


    return frame
last_alert_time = 0  # Store last alert timestamp
alert_interval = 600  # 10 minutes (600 seconds)

def send_crowd_alert(message):
    global last_alert_time
    current_time = time.time()
    
    if current_time - last_alert_time >= alert_interval:
        try:
            client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
            response = client.messages.create(
                body=message, 
                from_=TWILIO_PHONE, 
                to=ALERT_PHONE
            )
            last_alert_time = current_time  # Update last alert time
            print(f"Message Sent! SID: {response.sid}")
        except Exception as e:
            print(f"Error sending SMS: {e}")
    else:
        print("Skipping SMS: Cooldown active")

@app.route('/get_live_count')
def get_live_count():
    return jsonify({"people_count": current_people_count})

# Camera streaming route
@app.route('/stream')
def stream():
    def generate():
        cap = cv2.VideoCapture(0)  
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  
        cap.set(cv2.CAP_PROP_FPS, 30)  

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# UI Route
@app.route('/')
def index():
    return render_template('index.html', guest_verification=guest_verification_enabled, crowd_limit=crowd_limit)

# Toggle Guest Verification
@app.route('/toggle_guest_verification', methods=['POST'])
def toggle_guest_verification():
    global guest_verification_enabled
    guest_verification_enabled = not guest_verification_enabled
    return jsonify({"guest_verification": guest_verification_enabled})

# Set Perimeter
@app.route('/set_perimeter', methods=['POST'])
def set_perimeter():
    global monitoring_area
    data = request.get_json()
    if data and all(k in data for k in ["x1", "y1", "x2", "y2"]):
        monitoring_area = ((data['x1'], data['y1']), (data['x2'], data['y2']))
        return jsonify({"status": "Perimeter updated"})
    return jsonify({"error": "Invalid data"}), 400

@app.route('/get_people_count')
def get_people_count():
    global current_people_count
    return jsonify({"count": current_people_count})

# Set Crowd Limit
@app.route('/set_crowd_limit', methods=['POST'])
def set_crowd_limit():
    global crowd_limit
    data = request.json  # Ensure we're receiving JSON
    if 'limit' in data:
        crowd_limit = data['limit']
        return jsonify({"status": "Crowd limit updated", "new_limit": crowd_limit})
    return jsonify({"error": "Invalid data"}), 400

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'name' not in request.form:
        return jsonify({"message": "Missing image or name"}), 400
    
    image_file = request.files['image']
    person_name = request.form['name']

    # Read image and extract face encoding
    image = face_recognition.load_image_file(image_file)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        return jsonify({"message": "No face detected. Try another image."}), 400

    encoding_str = ','.join(map(str, encodings[0]))  # Convert encoding to string

    # Store in database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        clob = cursor.var(cx_Oracle.CLOB)  # Create CLOB variable
        clob.setvalue(0, encoding_str)  # Store encoding in CLOB

        cursor.execute("INSERT INTO face_encodings (name, encoding) VALUES (:1, TO_CLOB(:2))", (person_name, encoding_str))

        conn.commit()
        conn.close()

        return jsonify({"message": f"Face encoding for {person_name} stored successfully."})
    
    except cx_Oracle.DatabaseError as e:
        return jsonify({"message": "Database error occurred.", "error": str(e)}), 500

@app.route('/verify_guest', methods=['POST'])
def verify_guest():
    global guest_verification_enabled
    if not guest_verification_enabled:
        return jsonify({"message": "Guest verification is disabled."})

    frame = request.files['frame'].read()
    nparr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    unknown_encodings = face_recognition.face_encodings(img)
    
    if not unknown_encodings:
        return jsonify({"message": "No face detected."}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM face_encodings")
    stored_faces = cursor.fetchall()
    conn.close()

    known_encodings = []
    known_names = []

    for name, encoding_str in stored_faces:
        encoding_array = np.frombuffer(bytes(map(float, encoding_str.split(','))), dtype=np.float64)

        known_encodings.append(encoding_array)
        known_names.append(name)

    results = face_recognition.compare_faces(known_encodings, unknown_encodings[0])
    
    if True in results:
        match_index = results.index(True)
        return jsonify({"message": f"Guest recognized: {known_names[match_index]}", "status": "recognized", "color": "green"})
    else:
        return jsonify({"message": "Guest not recognized.", "status": "unknown", "color": "red"})
video_capture = cv2.VideoCapture(0)  # 0 for the default webcam, change if using an IP camera

@app.route('/guest_verification_stream')
def guest_verification_stream():
    return Response(generate_guest_verification_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def generate_guest_verification_frames():
    
    global guest_verification_enabled

    while True:
        success, frame = video_capture.read()
        if not success:
            continue
        

        if guest_verification_enabled:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, encoding FROM face_encodings")
            stored_faces = cursor.fetchall()
            conn.close()

            known_encodings = []
            known_names = []

            for name, encoding_str in stored_faces:
                encoding_array = np.fromstring(encoding_str, sep=',')
                known_encodings.append(encoding_array)
                known_names.append(name)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                color = (0, 0, 255)  # Default red for unrecognized faces

                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]
                    color = (0, 255, 0)  # Green for recognized guests

                # Draw rectangle and text
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/guest_verification')
def guest_verification_page():
    return render_template('guest_verification.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
