
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def register_user(email, password):
    return supabase.auth.sign_up({"email": email, "password": password})

def login_user(email, password):
    return supabase.auth.sign_in_with_password({"email": email, "password": password})

def get_user():
    return supabase.auth.get_user()

def is_user_premium(user_id):
    result = supabase.table("users").select("is_premium").eq("id", user_id).execute()
    if result.data and len(result.data) > 0:
        return result.data[0].get("is_premium", False)
    return False

