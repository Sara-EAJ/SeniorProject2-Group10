import os
import cv2
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime
from tkinter import scrolledtext

def format_xai_for_gui(xai):
    if xai:
        message = ""
        for r in xai.get("reasoning", []):
            message += f"{r}\n\n"
        return message
    return "No critical events detected. The AI monitoring system is operating normally."