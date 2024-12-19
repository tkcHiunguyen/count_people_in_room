# import requests
# import json
# import time
# import hashlib
# import hmac

# API_KEY = "ycns5nruw48pffjgwm4h"
# SECRET_KEY = "ec282ed931a24671a1da6195feef18e3"
# SWITCH_DEVICE_ID = "eb05ad32174c8c69564gik"
# BASE_URL = "https://openapi.tuyaus.com"

# def get_access_token():
#     timestamp = str(int(time.time() * 1000))
#     hash_str = hashlib.sha256(''.encode()).hexdigest()
#     sign_string = f"GET\n{hash_str}\n\n/v1.0/token?grant_type=1"
    
#     sign = hmac.new(
#         SECRET_KEY.encode('utf-8'),
#         (API_KEY + timestamp + sign_string).encode('utf-8'),
#         hashlib.sha256
#     ).hexdigest().upper()
    
#     headers = {
#         "Content-Type": "application/json",
#         "client_id": API_KEY,
#         "sign": sign,
#         "t": timestamp,
#         "sign_method": "HMAC-SHA256"
#     }
    
#     response = requests.get(f"{BASE_URL}/v1.0/token?grant_type=1", headers=headers)
#     result = response.json()
    
#     if not result.get('success', False):
#         raise Exception(f"Lỗi lấy token: {result.get('msg')}")
#     return result['result']['access_token']

# def turn_off_ac(access_token):
#     timestamp = str(int(time.time() * 1000))
#     command = {"commands": [{"code": "switch_1", "value": True}]}
#     body = json.dumps(command)
    
#     body_hash = hashlib.sha256(body.encode()).hexdigest()
#     sign_string = f"POST\n{body_hash}\n\n/v1.0/devices/{SWITCH_DEVICE_ID}/commands"
    
#     sign = hmac.new(
#         SECRET_KEY.encode('utf-8'),
#         (API_KEY + access_token + timestamp + sign_string).encode('utf-8'),
#         hashlib.sha256
#     ).hexdigest().upper()
    
#     headers = {
#         "Content-Type": "application/json",
#         "client_id": API_KEY,
#         "access_token": access_token,
#         "sign": sign,
#         "t": timestamp,
#         "sign_method": "HMAC-SHA256"
#     }
    
#     response = requests.post(
#         f"{BASE_URL}/v1.0/devices/{SWITCH_DEVICE_ID}/commands",
#         headers=headers,
#         data=body
#     )
#     result = response.json()
    
#     if result.get('success', False):
#         print("Đã tắt máy lạnh thành công!")
#     else:
#         print(f"Lỗi: {result.get('msg', 'Unknown error')}")

# if __name__ == "__main__":
#     try:
#         token = get_access_token()
#         turn_off_ac(token)
#     except Exception as e:
#         print(f"Lỗi: {str(e)}")
from tuya_connector import TuyaOpenAPI
API_KEY = "ycns5nruw48pffjgwm4h"
SECRET_KEY = "ec282ed931a24671a1da6195feef18e3"
SWITCH_DEVICE_ID = "eb05ad32174c8c69564gik"
BASE_URL = "https://openapi.tuyaus.com"


openapi = TuyaOpenAPI(BASE_URL,API_KEY,SECRET_KEY)
print( openapi.connect())
# /v1.0/iot-03/devices/eb05ad32174c8c69564gik/commands
value_controll="false"
command = {"commands":[{"code":"switch_1","value":value_controll}]}
openapi.post(f"/v1.0/iot-03/devices/{SWITCH_DEVICE_ID}/commands",command)