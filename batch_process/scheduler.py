import schedule
import time

from batch import run_batch


schedule.every().day.at("02:00").do(run_batch)
print("Batch scheduler started. Next run is scheduled for 02:00.")

while True:
    schedule.run_pending()
    time.sleep(30)

