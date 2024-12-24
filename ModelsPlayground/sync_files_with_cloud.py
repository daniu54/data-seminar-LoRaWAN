import os
import pickle
import subprocess
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]

# token is generated by code
TOKEN_PATH = "secrets/token.pickle"

# obtain using the tutorial here https://www.geeksforgeeks.org/get-list-of-files-and-folders-in-google-drive-storage-using-python/
CREDENTIALS_PATH = "secrets/credentials.json"

PIPELINE_FILE_LOCAL_PATH = "./pipeline.py"  # the file that contains the pipeline code
PIPELINE_FILE_GOOGLE_DRIVE_ID = "1-36OzBwKmd07w0sF9MYxu5d8fbLzo6_q"

MODEL_STATS_FILE_LOCAL_PATH = (
    "./results/model_stats.json"  # the file that contains the model stats
)
MODEL_STATS_FILE_GOOGLE_DRIVE_ID = "1-DsWqUI2PTrpb3fk3tGmOhOnKnsrbyMe"

GIT_STATUS_FILE_LOCAL_PATH = (
    "./results/git_status.txt"  # the file that contains the git status
)
GIT_STATUS_FILE_GOOGLE_DRIVE_ID = "1Qzuv3QmAHIhhU6Ue-Y9zJR5KA6IBSgdk"

creds = None


def connect_to_google_service():
    """
    Connects to Google Drive API and returns the service object.
    We need that to sync files with the Google Drive.
    """
    print("Connecting to Google Drive...")
    if os.path.exists(TOKEN_PATH):
        # Read the token from the file and
        # store it in the variable creds
        print("Reading token from", TOKEN_PATH)
        with open(TOKEN_PATH, "rb") as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        # If token is expired, it will be refreshed,
        # else, we will request a new one.
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing token...")
            creds.refresh(Request())
        else:
            print("Requesting new token from google server (will open browser)...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the access token in token.pickle
        # file for future usage
        with open(TOKEN_PATH, "wb") as token:
            print("Saving token to", TOKEN_PATH)
            pickle.dump(creds, token)

    service = build("drive", "v3", credentials=creds)
    print("Connected to Google Drive")

    return service


def get_file_list(
    file_count=None, query=None, fields="files(id, name, parents)", exclude_trashed=True
):
    """
    Get a list of files from Google Drive.
    Use fields parameter to specify which fields to include in the response.
    """
    service = connect_to_google_service()

    print("Getting files from Google Drive...")

    resource = service.files()

    if exclude_trashed:
        query += " and trashed=false"

    file_list = resource.list(pageSize=file_count, q=query, fields=fields).execute()
    file_list = file_list.get("files", [])

    print("Files retrieved successfully.")

    return file_list


def print_file_list(
    file_count=None, query=None, fields="files(id, name, parents)", exclude_trashed=True
):
    """
    Print all the files in the Google Drive.
    """
    file_list = get_file_list(
        file_count=file_count,
        query=query,
        fields=fields,
        exclude_trashed=exclude_trashed,
    )

    for file in file_list:
        print(file)


def update_file_in_cloud(file_path, file_id):
    """
    Update a file of the given Google Drive file_id with the file at file_path.
    """
    service = connect_to_google_service()

    print(
        f"Updating Google Drive file with ID '{file_id}' using file at {file_path}..."
    )

    request = service.files().get_media(fileId=file_id)

    media = MediaFileUpload(file_path, resumable=True)

    updated_file = service.files().update(fileId=file_id, media_body=media).execute()

    print(
        f"File '{updated_file['name']}' with id {updated_file['id']} was successfully updated."
    )

    return updated_file


def download_file_from_cloud(file_path, file_id):
    """
    Download a file from Google Drive with the given file_id to the file at file_path.
    """
    service = connect_to_google_service()

    print(
        f"Updating Google Drive file with ID '{file_id}' using file at {file_path}..."
    )

    media = MediaFileUpload(file_path, resumable=True)

    request = service.files().get_media(fileId=file_id)

    file_metadata = {}

    with open(file_path, "wb") as file:
        downloader = MediaIoBaseDownload(file, request)

        done = False

        while not done:
            status, done = downloader.next_chunk()
            print(f"Download progress: {int(status.progress() * 100)}%")

    print(f"File downloaded successfully to '{file_path}'.")


def save_git_commit_info_to_file(file_path):
    try:
        # Get the current Git commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        print(f"Current Git commit hash: {commit_hash}")

        # Write the commit hash to a local file
        with open(file_path, "w") as file:
            file.write(f"Commit Hash: {commit_hash}\n")
        print(f"Commit information written to {file_path}")

    except subprocess.CalledProcessError as e:
        print("Error fetching Git commit hash. Are you in a Git repository?")
        raise e
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    download_file_from_cloud(
        MODEL_STATS_FILE_LOCAL_PATH, MODEL_STATS_FILE_GOOGLE_DRIVE_ID
    )

    update_file_in_cloud(PIPELINE_FILE_LOCAL_PATH, PIPELINE_FILE_GOOGLE_DRIVE_ID)

    save_git_commit_info_to_file(GIT_STATUS_FILE_LOCAL_PATH)
    update_file_in_cloud(GIT_STATUS_FILE_LOCAL_PATH, GIT_STATUS_FILE_GOOGLE_DRIVE_ID)
