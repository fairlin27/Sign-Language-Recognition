import pyttsx3

engine = pyttsx3.init()

engine.setProperty("rate", 150)

engine.setProperty("volume", 1.0)

print("Speaking...")

engine.say("Hello")

engine.runAndWait()

print("Finished")