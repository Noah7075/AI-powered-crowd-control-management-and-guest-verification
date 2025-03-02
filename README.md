📌 Overview
The AI-Powered Crowd Monitoring System is designed for real-time crowd detection and monitoring using computer vision and deep learning. It utilizes OpenCV, Dlib, YOLO, DeepSORT, and TensorFlow/PyTorch to analyze crowd density and identify individuals. The system can send Twilio SMS alerts when a large crowd is detected and supports guest verification using face recognition.

🚀 Features
✅ Real-Time Crowd Monitoring – Detects and tracks people using a CCTV or IP camera.
✅ Guest Verification – Matches detected faces with stored face encodings in SQLPlus.
✅ Admin Dashboard – Manage attendance records, upload images, and configure system settings.
✅ Alert System – Sends SMS alerts via Twilio when a large crowd is detected.
✅ Efficient Camera Reconnection – Ensures continuous streaming even if the connection drops.
✅ Improved Database Handling – Prevents crashes due to database errors.
✅ Optimized Performance – Prevents frequent SMS spam by adding an alert cooldown.
✅ Debugging Support – Added logging to track errors and system activity.

🛠️ Installation & Setup
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/Crowd-Monitoring-App.git
cd Crowd-Monitoring-App
2️⃣ Install Required Dependencies
Ensure you have Python installed, then run:

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Application
bash
Copy
Edit
python app.py
4️⃣ Access the Web Interface
Once the app is running, open your browser and go to:

arduino
Copy
Edit
http://localhost:5000/
⚙️ Tech Stack
Backend: Flask / FastAPI
AI Models: OpenCV, Dlib, YOLO, DeepSORT
Database: MySQL / PostgreSQL / SQLPlus
Alerts: Twilio API
Deployment: Can be hosted on Heroku, AWS, or DigitalOcean
🛠️ Environment Variables
To configure your database and Twilio settings, set the following environment variables:

makefile
Copy
Edit
DB_HOST=
DB_USER=
DB_PASSWORD=
DB_NAME=
⚠ Note: Twilio credentials are hardcoded as per project requirements.

📢 Contributing
Pull requests are welcome! If you want to contribute, please fork the repository and create a new branch for your changes.

📜 License
This project is open-source and available under the MIT License.

