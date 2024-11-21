import os
from pathlib import Path
import shutil

class ExternalDriveManager:
    def __init__(self, drive_name="Crucial X9"):
        self.drive_path = Path("/Volumes") / drive_name
        self.check_drive_mounted()
    
    def check_drive_mounted(self):
        """Verify that the drive is mounted and accessible"""
        if not self.drive_path.exists():
            raise RuntimeError(
                f"Drive not found at {self.drive_path}. "
                "Please ensure the drive is connected and mounted."
            )
    
    def list_contents(self, subpath=""):
        """
        List contents of the drive or a specific subdirectory
        
        Parameters:
        subpath (str): Optional subdirectory path within the drive
        """
        target_path = self.drive_path / subpath
        print(f"\nContents of {target_path}:")
        for item in target_path.iterdir():
            size = item.stat().st_size if item.is_file() else '<DIR>'
            print(f"{'File' if item.is_file() else 'Dir'}: {item.name} - Size: {size} bytes")
    
    def create_directory(self, dir_name):
        """Create a new directory on the drive"""
        new_dir = self.drive_path / dir_name
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir
    
    def copy_to_drive(self, source_path, destination_name=None):
        """
        Copy a file to the drive
        
        Parameters:
        source_path (str): Path to the source file
        destination_name (str): Optional new name for the file
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file {source_path} not found")
            
        dest_name = destination_name or source.name
        destination = self.drive_path / dest_name
        
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            shutil.copytree(source, destination)
        return destination
    
    def copy_from_drive(self, filename, destination_path):
        """Copy a file from the drive to the local system"""
        source = self.drive_path / filename
        destination = Path(destination_path)
        
        if not source.exists():
            raise FileNotFoundError(f"File {filename} not found on drive")
            
        if source.is_file():
            shutil.copy2(source, destination)
        else:
            shutil.copytree(source, destination)
        return destination
    
    def search_files(self, pattern):
        """Search for files matching a pattern"""
        return list(self.drive_path.glob(pattern))
    
    def get_drive_info(self):
        """Get basic information about the drive"""
        total, used, free = shutil.disk_usage(self.drive_path)
        return {
            'total': total // (2**30),  # Convert to GB
            'used': used // (2**30),
            'free': free // (2**30)
        }
        
drive = ExternalDriveManager("Crucial X9")

info = drive.get_drive_info()
print(f"Drive space: {info['free']}GB free of {info['total']}GB")

drive.list_contents()
