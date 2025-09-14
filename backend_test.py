import requests
import sys
import json
from datetime import datetime, timedelta
import time

class TeleMedProAPITester:
    def __init__(self, base_url="https://careconnect-76.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.patient_id = None
        self.session_id = None
        self.doctor_id = None
        self.prescription_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}" if endpoint else f"{self.api_url}/"
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, data=data, files=files, timeout=30)
                else:
                    response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=30)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and len(str(response_data)) < 500:
                        print(f"   Response: {response_data}")
                    return True, response_data
                except:
                    return True, response.text
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data}")
                except:
                    print(f"   Error: {response.text}")
                return False, {}

        except requests.exceptions.Timeout:
            print(f"❌ Failed - Request timeout (30s)")
            return False, {}
        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        return self.run_test("Root API Endpoint", "GET", "", 200)

    def test_create_patient(self):
        """Test patient creation"""
        patient_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "1234567890",
            "age": 35,
            "gender": "male",
            "language_preference": "en"
        }
        
        success, response = self.run_test(
            "Create Patient",
            "POST",
            "patients",
            200,
            data=patient_data
        )
        
        if success and 'id' in response:
            self.patient_id = response['id']
            print(f"   Patient ID: {self.patient_id}")
            return True
        return False

    def test_get_patient(self):
        """Test getting patient by ID"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False
            
        return self.run_test(
            "Get Patient",
            "GET",
            f"patients/{self.patient_id}",
            200
        )[0]

    def test_transcribe_text(self):
        """Test text transcription"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False

        form_data = {
            'patient_id': self.patient_id,
            'language': 'en',
            'text': 'I have been experiencing headache and mild fever for the past 2 days'
        }
        
        return self.run_test(
            "Transcribe Text Input",
            "POST",
            "transcribe",
            200,
            data=form_data,
            files={}  # This indicates form data
        )[0]

    def test_symptom_assessment_mild(self):
        """Test AI symptom assessment with mild symptoms"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False

        symptom_data = {
            "patient_id": self.patient_id,
            "symptoms": "I have a slight headache and runny nose for 1 day",
            "language": "en"
        }
        
        print("   Testing mild symptoms - this may take 10-15 seconds for AI processing...")
        success, response = self.run_test(
            "AI Assessment - Mild Symptoms",
            "POST",
            "assess",
            200,
            data=symptom_data
        )
        
        if success and 'id' in response:
            self.session_id = response['id']
            print(f"   Session ID: {self.session_id}")
            if 'triage_result' in response:
                triage = response['triage_result']
                print(f"   Triage Level: {triage.get('triage_level', 'N/A')}")
                print(f"   Urgency Score: {triage.get('urgency_score', 'N/A')}/10")
            return True
        return False

    def test_symptom_assessment_moderate(self):
        """Test AI symptom assessment with moderate symptoms"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False

        symptom_data = {
            "patient_id": self.patient_id,
            "symptoms": "I have fever of 101°F, body aches, and cough for 3 days",
            "language": "en"
        }
        
        print("   Testing moderate symptoms - this may take 10-15 seconds for AI processing...")
        success, response = self.run_test(
            "AI Assessment - Moderate Symptoms",
            "POST",
            "assess",
            200,
            data=symptom_data
        )
        
        if success and 'triage_result' in response:
            triage = response['triage_result']
            print(f"   Triage Level: {triage.get('triage_level', 'N/A')}")
            print(f"   Urgency Score: {triage.get('urgency_score', 'N/A')}/10")
            return True
        return False

    def test_symptom_assessment_severe(self):
        """Test AI symptom assessment with severe symptoms"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False

        symptom_data = {
            "patient_id": self.patient_id,
            "symptoms": "I have severe chest pain, difficulty breathing, and sweating",
            "language": "en"
        }
        
        print("   Testing severe symptoms - this may take 10-15 seconds for AI processing...")
        success, response = self.run_test(
            "AI Assessment - Severe Symptoms",
            "POST",
            "assess",
            200,
            data=symptom_data
        )
        
        if success and 'triage_result' in response:
            triage = response['triage_result']
            print(f"   Triage Level: {triage.get('triage_level', 'N/A')}")
            print(f"   Urgency Score: {triage.get('urgency_score', 'N/A')}/10")
            return True
        return False

    def test_get_doctors(self):
        """Test getting available doctors"""
        success, response = self.run_test(
            "Get Available Doctors",
            "GET",
            "doctors",
            200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            self.doctor_id = response[0]['id']
            print(f"   Found {len(response)} doctors")
            print(f"   First doctor ID: {self.doctor_id}")
            return True
        return False

    def test_doctor_queue(self):
        """Test getting doctor queue"""
        return self.run_test(
            "Get Doctor Queue",
            "GET",
            "doctor/queue",
            200
        )[0]

    def test_assign_doctor(self):
        """Test assigning doctor to session"""
        if not self.session_id or not self.doctor_id:
            print("❌ Skipping - No session ID or doctor ID available")
            return False

        assignment_data = {
            "session_id": self.session_id,
            "doctor_id": self.doctor_id
        }
        
        return self.run_test(
            "Assign Doctor to Session",
            "POST",
            "doctor/assign",
            200,
            data=assignment_data
        )[0]

    def test_send_chat_message(self):
        """Test sending chat message"""
        if not self.session_id or not self.patient_id:
            print("❌ Skipping - No session ID or patient ID available")
            return False

        form_data = {
            'session_id': self.session_id,
            'sender_type': 'patient',
            'sender_id': self.patient_id,
            'message': 'Hello doctor, I need help with my symptoms.'
        }
        
        return self.run_test(
            "Send Chat Message",
            "POST",
            "chat/send",
            200,
            data=form_data,
            files={}
        )[0]

    def test_get_chat_messages(self):
        """Test getting chat messages"""
        if not self.session_id:
            print("❌ Skipping - No session ID available")
            return False
            
        return self.run_test(
            "Get Chat Messages",
            "GET",
            f"chat/{self.session_id}",
            200
        )[0]

    def test_create_prescription(self):
        """Test creating digital prescription"""
        if not self.session_id:
            print("❌ Skipping - No session ID available")
            return False

        prescription_data = {
            "session_id": self.session_id,
            "medications": [
                {
                    "name": "Paracetamol",
                    "dosage": "500mg",
                    "frequency": "Twice daily",
                    "duration": "5 days",
                    "quantity": 10,
                    "price": 25.0
                }
            ],
            "instructions": "Take with food. Complete the full course.",
            "validity_days": 30
        }
        
        success, response = self.run_test(
            "Create Digital Prescription",
            "POST",
            "prescriptions",
            200,
            data=prescription_data
        )
        
        if success and 'id' in response:
            self.prescription_id = response['id']
            print(f"   Prescription ID: {self.prescription_id}")
            return True
        return False

    def test_get_patient_prescriptions(self):
        """Test getting patient prescriptions"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False
            
        return self.run_test(
            "Get Patient Prescriptions",
            "GET",
            f"prescriptions/patient/{self.patient_id}",
            200
        )[0]

    def test_book_lab_test(self):
        """Test booking lab test"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False

        booking_data = {
            "patient_id": self.patient_id,
            "session_id": self.session_id,
            "test_name": "Complete Blood Count (CBC)",
            "lab_provider": "HealthLab Diagnostics",
            "preferred_date": (datetime.now() + timedelta(days=2)).isoformat()
        }
        
        return self.run_test(
            "Book Lab Test",
            "POST",
            "lab/book",
            200,
            data=booking_data
        )[0]

    def test_get_patient_lab_bookings(self):
        """Test getting patient lab bookings"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False
            
        return self.run_test(
            "Get Patient Lab Bookings",
            "GET",
            f"lab/patient/{self.patient_id}",
            200
        )[0]

    def test_place_pharmacy_order(self):
        """Test placing pharmacy order"""
        if not self.patient_id or not self.prescription_id:
            print("❌ Skipping - No patient ID or prescription ID available")
            return False

        order_data = {
            "patient_id": self.patient_id,
            "prescription_id": self.prescription_id,
            "pharmacy_name": "MedPlus Pharmacy",
            "delivery_address": "123 Main St, City, State 12345"
        }
        
        return self.run_test(
            "Place Pharmacy Order",
            "POST",
            "pharmacy/order",
            200,
            data=order_data
        )[0]

    def test_get_patient_pharmacy_orders(self):
        """Test getting patient pharmacy orders"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False
            
        return self.run_test(
            "Get Patient Pharmacy Orders",
            "GET",
            f"pharmacy/patient/{self.patient_id}",
            200
        )[0]

    def test_patient_dashboard(self):
        """Test patient dashboard"""
        if not self.patient_id:
            print("❌ Skipping - No patient ID available")
            return False
            
        return self.run_test(
            "Get Patient Dashboard",
            "GET",
            f"dashboard/patient/{self.patient_id}",
            200
        )[0]

def main():
    print("🏥 TeleMed-Pro Backend API Testing")
    print("=" * 50)
    
    tester = TeleMedProAPITester()
    
    # Test sequence
    test_functions = [
        tester.test_root_endpoint,
        tester.test_create_patient,
        tester.test_get_patient,
        tester.test_transcribe_text,
        tester.test_symptom_assessment_mild,
        tester.test_symptom_assessment_moderate,
        tester.test_symptom_assessment_severe,
        tester.test_get_doctors,
        tester.test_doctor_queue,
        tester.test_assign_doctor,
        tester.test_send_chat_message,
        tester.test_get_chat_messages,
        tester.test_create_prescription,
        tester.test_get_patient_prescriptions,
        tester.test_book_lab_test,
        tester.test_get_patient_lab_bookings,
        tester.test_place_pharmacy_order,
        tester.test_get_patient_pharmacy_orders,
        tester.test_patient_dashboard
    ]
    
    # Run all tests
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Print results
    print("\n" + "=" * 50)
    print(f"📊 BACKEND TEST RESULTS")
    print(f"Tests passed: {tester.tests_passed}/{tester.tests_run}")
    print(f"Success rate: {(tester.tests_passed/tester.tests_run)*100:.1f}%")
    
    if tester.tests_passed == tester.tests_run:
        print("🎉 All backend tests passed!")
        return 0
    else:
        print("⚠️  Some backend tests failed. Check the details above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())