"""Example: read raw news and produce a snapshot signal."""
from services.news_signal import NewsSignalService

if __name__ == "__main__":
    svc = NewsSignalService()
    snapshot = svc.process_recent(days=2)
    print("Snapshot:")
    print(snapshot)
