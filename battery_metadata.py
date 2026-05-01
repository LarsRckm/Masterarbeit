import os

def extract_cell_format(format_dir_name):
    """
    Extrahierte das Zellformat (z.B. 4680, 2170, 18650) aus dem Ordnernamen.
    Splittet nach '_' falls vorhanden.
    """
    if '_' in format_dir_name:
        return format_dir_name.split('_')[0]
    return format_dir_name

def get_max_height_from_format(cell_format):
    """
    Bestimmt die maximale Höhe aus dem Zellformat (z.B. 18650 -> 65, 2170 -> 70, 4680 -> 80).
    Die ersten zwei Zahlen stehen für den Durchmesser, die nächsten zwei für die Höhe.
    Falls eine letzte Zahl existiert (wie die 0 bei 18650), wird diese vernachlässigt.
    """
    if len(cell_format) >= 4:
        return float(cell_format[2:4])
    return None

def extract_absolute_depth(filename):
    """
    Extrahiert die absolute Slice-Tiefe aus dem Dateinamen.
    Alle Dateien enden auf .png und haben die Tiefe davor stehen.
    """
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split('_')
    try:
        # Nehme den letzten Teil des Dateinamens als absolute Tiefe an
        return float(parts[-1])
    except ValueError:
        return 0.0

def determine_manufacturer(width, height, relative_path):
    """
    Bestimmt den Hersteller anhand der Pixelanzahl (Bildgröße) und des Pfads.
    """
    if width == 1340 and height == 1340:
        return "Samsung"
    elif width == 1370 and height == 1370:
        return "Vapcell"
    elif width == 1342 and height == 1342:
        return "BYD"
    elif width == 1320 and height == 1320:
        # Bei 1320x1320 (18650) im Pfad nach 'sodium' schauen
        if "sodium" in relative_path.lower():
            return "HAKADI"
        else:
            return "EVE"
    else:
        return "Unknown"

def determine_chemistry(manufacturer):
    """
    Bestimmt die Batterie-Chemie basierend auf dem Hersteller.
    (Behebt den Fehler im Bild: Hakadi ist Sodium-ion)
    """
    if manufacturer in ["HAKADI", "Vapcell"]:
        return "Sodium-ion"
    else:
        return "Lithium-ion"

def determine_voxel_size(cell_format):
    """
    Bestimmt die Voxelgröße in µm basierend auf dem Zellformat.
    """
    if "18650" in cell_format:
        return 14.4
    elif "2170" in cell_format:
        return 16.4
    elif "4680" in cell_format:
        return 35.0
    return None
