from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
from emergentintegrations.llm.chat import LlmChat, UserMessage
import json
import asyncio

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="TeleMed-Pro API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DATA MODELS ====================

class Patient(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    phone: str
    age: int
    gender: str
    language_preference: str = "en"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Doctor(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    email: str
    specialization: str
    license_number: str
    available: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class TriageAssessment(BaseModel):
    triage_level: str  # self-care, primary-care, urgent, emergency
    probable_conditions: List[str] = []
    red_flags: List[str] = []
    recommended_actions: List[str] = []
    urgency_score: int  # 1-10 scale
    explanation: str
    confidence_level: float  # 0-1 scale

class ConsultationSession(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    doctor_id: Optional[str] = None
    symptoms: str
    audio_transcript: Optional[str] = None
    triage_result: Optional[TriageAssessment] = None
    status: str = "pending"  # pending, in_progress, completed, cancelled
    priority: str = "normal"  # low, normal, high, critical
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    sender_type: str  # patient, doctor, system
    sender_id: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DigitalPrescription(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    doctor_id: str
    patient_id: str
    medications: List[Dict[str, Any]]
    instructions: str
    valid_till: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LabBooking(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    session_id: Optional[str] = None
    test_name: str
    lab_provider: str
    scheduled_date: datetime
    status: str = "scheduled"  # scheduled, sample_collected, processing, completed
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class PharmacyOrder(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str
    prescription_id: str
    pharmacy_name: str
    medications: List[Dict[str, Any]]
    total_amount: float
    status: str = "placed"  # placed, confirmed, prepared, delivered
    delivery_address: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ==================== REQUEST/RESPONSE MODELS ====================

class PatientCreate(BaseModel):
    name: str
    email: str
    phone: str
    age: int
    gender: str
    language_preference: str = "en"

class SymptomInput(BaseModel):
    patient_id: str
    symptoms: str
    language: str = "en"
    urgency_indicated: Optional[bool] = False

class TranscribeRequest(BaseModel):
    patient_id: str
    language: str = "en"

class DoctorAssignment(BaseModel):
    session_id: str
    doctor_id: str

class PrescriptionCreate(BaseModel):
    session_id: str
    medications: List[Dict[str, Any]]
    instructions: str
    validity_days: int = 30

class LabBookingCreate(BaseModel):
    patient_id: str
    session_id: Optional[str] = None
    test_name: str
    lab_provider: str
    preferred_date: str

class PharmacyOrderCreate(BaseModel):
    patient_id: str
    prescription_id: str
    pharmacy_name: str
    delivery_address: str

# ==================== AI TRIAGE SYSTEM ====================

class MedicalTriageAI:
    def __init__(self):
        self.llm_chat = LlmChat(
            api_key=os.environ.get('EMERGENT_LLM_KEY'),
            session_id="medical_triage",
            system_message="""You are a conservative medical triage AI assistant. 
            
Your role is to assess patient symptoms and provide structured triage recommendations.

IMPORTANT GUIDELINES:
1. Never provide definitive diagnoses
2. Always err on the side of caution
3. Escalate anything that could be serious
4. Include appropriate disclaimers
5. Focus on urgency and next steps, not specific treatments

For each assessment, analyze:
- Symptom severity and combination
- Potential red flags requiring immediate attention
- Appropriate triage level
- Recommended next steps

RED FLAGS that require EMERGENCY triage:
- Chest pain with breathing difficulty
- Loss of consciousness
- Severe bleeding
- Signs of stroke (sudden weakness, speech problems)
- Severe abdominal pain
- High fever with severe symptoms
- Breathing difficulties
- Allergic reactions

Return responses in this exact JSON format:
{
  "triage_level": "emergency|urgent|primary-care|self-care",
  "probable_conditions": ["condition1", "condition2"],
  "red_flags": ["flag1", "flag2"],
  "recommended_actions": ["action1", "action2"],
  "urgency_score": 1-10,
  "explanation": "Clear explanation of assessment",
  "confidence_level": 0.0-1.0
}

Always include disclaimer: "This is an AI assessment, not a medical diagnosis. Consult healthcare professionals."
            """
        ).with_model("openai", "gpt-4o")

    async def assess_symptoms(self, symptoms: str, patient_context: Dict = None) -> TriageAssessment:
        try:
            context_str = ""
            if patient_context:
                context_str = f"\nPatient context: Age {patient_context.get('age', 'unknown')}, Gender {patient_context.get('gender', 'unknown')}"
            
            user_message = UserMessage(
                text=f"Please assess these symptoms: {symptoms}{context_str}\n\nProvide structured JSON response following the exact format specified."
            )
            
            # Get AI response
            response = await self.llm_chat.send_message(user_message)
            
            # Parse JSON response
            try:
                # Extract JSON from response if it's wrapped in markdown or other text
                response_text = response
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.rfind("```")
                    response_text = response_text[json_start:json_end].strip()
                
                assessment_data = json.loads(response_text)
                return TriageAssessment(**assessment_data)
                
            except json.JSONDecodeError:
                # Fallback assessment if JSON parsing fails
                logger.error(f"Failed to parse AI response: {response}")
                return TriageAssessment(
                    triage_level="primary-care",
                    probable_conditions=["Assessment needed"],
                    red_flags=[],
                    recommended_actions=["Consult healthcare provider"],
                    urgency_score=5,
                    explanation="AI assessment temporarily unavailable. Please consult a healthcare provider for proper evaluation.",
                    confidence_level=0.5
                )
                
        except Exception as e:
            logger.error(f"AI triage assessment failed: {str(e)}")
            # Return conservative fallback
            return TriageAssessment(
                triage_level="urgent",
                probable_conditions=["Requires medical evaluation"],
                red_flags=["Assessment system temporarily unavailable"],
                recommended_actions=["Seek immediate medical consultation"],
                urgency_score=7,
                explanation="Unable to process assessment. Please seek immediate medical advice for safety.",
                confidence_level=0.3
            )

# Initialize AI system
triage_ai = MedicalTriageAI()

# ==================== API ENDPOINTS ====================

@api_router.get("/")
async def root():
    return {"message": "TeleMed-Pro API", "version": "1.0.0", "status": "active"}

# ==================== PATIENT MANAGEMENT ====================

@api_router.post("/patients", response_model=Patient)
async def create_patient(patient_data: PatientCreate):
    patient = Patient(**patient_data.dict())
    await db.patients.insert_one(patient.dict())
    return patient

@api_router.get("/patients/{patient_id}", response_model=Patient)
async def get_patient(patient_id: str):
    patient = await db.patients.find_one({"id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return Patient(**patient)

# ==================== SYMPTOM ASSESSMENT & TRIAGE ====================

@api_router.post("/transcribe")
async def transcribe_audio(
    patient_id: str = Form(...),
    language: str = Form("en"),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """Transcribe audio or accept text input for symptom assessment"""
    try:
        if text:
            # Direct text input
            transcript = text
        elif audio:
            # For MVP, we'll simulate transcription
            # In production, you'd use speech recognition here
            transcript = "Patient provided audio input - transcription would be processed here"
        else:
            raise HTTPException(status_code=400, detail="Either text or audio input required")
        
        return {
            "transcript": transcript,
            "language": language,
            "patient_id": patient_id,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(status_code=500, detail="Transcription failed")

@api_router.post("/assess", response_model=ConsultationSession)
async def assess_symptoms(symptom_data: SymptomInput):
    """AI-powered symptom assessment and triage"""
    try:
        # Get patient context
        patient = await db.patients.find_one({"id": symptom_data.patient_id})
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        patient_context = {
            "age": patient.get("age"),
            "gender": patient.get("gender")
        }
        
        # Perform AI triage assessment
        triage_result = await triage_ai.assess_symptoms(
            symptom_data.symptoms, 
            patient_context
        )
        
        # Determine priority based on triage level
        priority_mapping = {
            "emergency": "critical",
            "urgent": "high", 
            "primary-care": "normal",
            "self-care": "low"
        }
        
        # Create consultation session
        session = ConsultationSession(
            patient_id=symptom_data.patient_id,
            symptoms=symptom_data.symptoms,
            triage_result=triage_result,
            status="pending",
            priority=priority_mapping.get(triage_result.triage_level, "normal")
        )
        
        # Save to database
        await db.consultation_sessions.insert_one(session.dict())
        
        return session
        
    except Exception as e:
        logger.error(f"Symptom assessment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Assessment failed")

# ==================== DOCTOR MANAGEMENT ====================

@api_router.get("/doctors", response_model=List[Doctor])
async def get_available_doctors():
    """Get list of available doctors"""
    doctors = await db.doctors.find({"available": True}).to_list(100)
    return [Doctor(**doc) for doc in doctors]

@api_router.get("/doctor/queue")
async def get_doctor_queue():
    """Get pending consultation sessions for doctors"""
    sessions = await db.consultation_sessions.find({
        "status": "pending"
    }).sort("created_at", 1).to_list(50)
    
    return {
        "queue": [ConsultationSession(**session) for session in sessions],
        "total_pending": len(sessions)
    }

@api_router.post("/doctor/assign")
async def assign_doctor(assignment: DoctorAssignment):
    """Assign doctor to consultation session"""
    try:
        # Update session with doctor assignment
        result = await db.consultation_sessions.update_one(
            {"id": assignment.session_id},
            {
                "$set": {
                    "doctor_id": assignment.doctor_id,
                    "status": "in_progress",
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {"status": "success", "message": "Doctor assigned successfully"}
    except Exception as e:
        logger.error(f"Doctor assignment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Assignment failed")

# ==================== CONSULTATION CHAT ====================

@api_router.post("/chat/send")
async def send_message(
    session_id: str = Form(...),
    sender_type: str = Form(...),
    sender_id: str = Form(...),
    message: str = Form(...)
):
    """Send message in consultation chat"""
    try:
        chat_message = ChatMessage(
            session_id=session_id,
            sender_type=sender_type,
            sender_id=sender_id,
            message=message
        )
        
        await db.chat_messages.insert_one(chat_message.dict())
        return chat_message
    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        raise HTTPException(status_code=500, detail="Message failed")

@api_router.get("/chat/{session_id}")
async def get_chat_messages(session_id: str):
    """Get chat messages for a session"""
    messages = await db.chat_messages.find({
        "session_id": session_id
    }).sort("timestamp", 1).to_list(100)
    
    return {
        "messages": [ChatMessage(**msg) for msg in messages],
        "session_id": session_id
    }

# ==================== DIGITAL PRESCRIPTIONS ====================

@api_router.post("/prescriptions", response_model=DigitalPrescription)
async def create_prescription(prescription_data: PrescriptionCreate):
    """Create digital prescription"""
    try:
        # Get session details
        session = await db.consultation_sessions.find_one({"id": prescription_data.session_id})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Create prescription
        valid_till = datetime.now(timezone.utc) + timedelta(days=prescription_data.validity_days)
        
        prescription = DigitalPrescription(
            session_id=prescription_data.session_id,
            doctor_id=session["doctor_id"],
            patient_id=session["patient_id"],
            medications=prescription_data.medications,
            instructions=prescription_data.instructions,
            valid_till=valid_till
        )
        
        await db.prescriptions.insert_one(prescription.dict())
        return prescription
    except Exception as e:
        logger.error(f"Prescription creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prescription creation failed")

@api_router.get("/prescriptions/patient/{patient_id}")
async def get_patient_prescriptions(patient_id: str):
    """Get prescriptions for a patient"""
    prescriptions = await db.prescriptions.find({
        "patient_id": patient_id
    }).sort("created_at", -1).to_list(20)
    
    return {
        "prescriptions": [DigitalPrescription(**p) for p in prescriptions]
    }

# ==================== LAB BOOKING ====================

@api_router.post("/lab/book", response_model=LabBooking)
async def book_lab_test(booking_data: LabBookingCreate):
    """Book lab test"""
    try:
        from datetime import datetime
        scheduled_date = datetime.fromisoformat(booking_data.preferred_date)
        
        lab_booking = LabBooking(
            patient_id=booking_data.patient_id,
            session_id=booking_data.session_id,
            test_name=booking_data.test_name,
            lab_provider=booking_data.lab_provider,
            scheduled_date=scheduled_date
        )
        
        await db.lab_bookings.insert_one(lab_booking.dict())
        return lab_booking
    except Exception as e:
        logger.error(f"Lab booking error: {str(e)}")
        raise HTTPException(status_code=500, detail="Lab booking failed")

@api_router.get("/lab/patient/{patient_id}")
async def get_patient_lab_bookings(patient_id: str):
    """Get lab bookings for a patient"""
    bookings = await db.lab_bookings.find({
        "patient_id": patient_id
    }).sort("scheduled_date", -1).to_list(20)
    
    return {
        "bookings": [LabBooking(**b) for b in bookings]
    }

# ==================== PHARMACY INTEGRATION ====================

@api_router.post("/pharmacy/order", response_model=PharmacyOrder)
async def place_pharmacy_order(order_data: PharmacyOrderCreate):
    """Place pharmacy order"""
    try:
        # Get prescription details
        prescription = await db.prescriptions.find_one({"id": order_data.prescription_id})
        if not prescription:
            raise HTTPException(status_code=404, detail="Prescription not found")
        
        # Calculate total amount (mock calculation)
        total_amount = sum(med.get("price", 50) * med.get("quantity", 1) 
                          for med in prescription["medications"])
        
        pharmacy_order = PharmacyOrder(
            patient_id=order_data.patient_id,
            prescription_id=order_data.prescription_id,
            pharmacy_name=order_data.pharmacy_name,
            medications=prescription["medications"],
            total_amount=total_amount,
            delivery_address=order_data.delivery_address
        )
        
        await db.pharmacy_orders.insert_one(pharmacy_order.dict())
        return pharmacy_order
    except Exception as e:
        logger.error(f"Pharmacy order error: {str(e)}")
        raise HTTPException(status_code=500, detail="Pharmacy order failed")

@api_router.get("/pharmacy/patient/{patient_id}")
async def get_patient_pharmacy_orders(patient_id: str):
    """Get pharmacy orders for a patient"""
    orders = await db.pharmacy_orders.find({
        "patient_id": patient_id
    }).sort("created_at", -1).to_list(20)
    
    return {
        "orders": [PharmacyOrder(**o) for o in orders]
    }

# ==================== DASHBOARD ENDPOINTS ====================

@api_router.get("/dashboard/patient/{patient_id}")
async def get_patient_dashboard(patient_id: str):
    """Get patient dashboard data"""
    try:
        # Get recent sessions
        sessions = await db.consultation_sessions.find({
            "patient_id": patient_id
        }).sort("created_at", -1).limit(5).to_list(5)
        
        # Get recent prescriptions
        prescriptions = await db.prescriptions.find({
            "patient_id": patient_id
        }).sort("created_at", -1).limit(3).to_list(3)
        
        # Get upcoming lab bookings
        upcoming_labs = await db.lab_bookings.find({
            "patient_id": patient_id,
            "status": {"$ne": "completed"}
        }).sort("scheduled_date", 1).limit(3).to_list(3)
        
        return {
            "recent_sessions": [ConsultationSession(**s) for s in sessions],
            "recent_prescriptions": [DigitalPrescription(**p) for p in prescriptions],
            "upcoming_labs": [LabBooking(**l) for l in upcoming_labs]
        }
    except Exception as e:
        logger.error(f"Dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail="Dashboard data failed")

# Include the router in the main app
app.include_router(api_router)

# ==================== STARTUP EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize database with sample data"""
    # Create sample doctors if none exist
    doctor_count = await db.doctors.count_documents({})
    if doctor_count == 0:
        sample_doctors = [
            {
                "id": str(uuid.uuid4()),
                "name": "Dr. Priya Sharma",
                "email": "priya.sharma@telemed.com",
                "specialization": "General Medicine",
                "license_number": "MED12345",
                "available": True,
                "created_at": datetime.now(timezone.utc)
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Dr. Rajesh Kumar",
                "email": "rajesh.kumar@telemed.com", 
                "specialization": "Cardiology",
                "license_number": "MED12346",
                "available": True,
                "created_at": datetime.now(timezone.utc)
            },
            {
                "id": str(uuid.uuid4()),
                "name": "Dr. Anita Patel",
                "email": "anita.patel@telemed.com",
                "specialization": "Dermatology", 
                "license_number": "MED12347",
                "available": True,
                "created_at": datetime.now(timezone.utc)
            }
        ]
        await db.doctors.insert_many(sample_doctors)
        logger.info("Sample doctors created")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()