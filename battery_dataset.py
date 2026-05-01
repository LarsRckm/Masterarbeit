import os
import random
from PIL import Image
from torch.utils.data import IterableDataset

try:
    from model import config
    from model import battery_metadata
except ImportError:
    import config
    import battery_metadata

class BatteryCTDataset(IterableDataset):
    def __init__(self, base_path=None, min_rel_depth=0.1, max_rel_depth=0.9):
        """
        Dataset, das ein Bild und die zugehörigen Bedingungen lädt.
        """
        super().__init__()
        self.base_path = base_path if base_path is not None else config.BASE_PATH
        self.min_rel_depth = min_rel_depth
        self.max_rel_depth = max_rel_depth
        
        self.cell_formats = []
        self.format_to_cells = {}
        
        # Durchsuche die Ordnerstruktur initial
        if os.path.exists(self.base_path):
            format_dirs = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
            
            for format_dir in format_dirs:
                cell_format = battery_metadata.extract_cell_format(format_dir)
                
                # Wir gehen davon aus, dass wir nur 18650, 2170, 4680 betrachten
                if not any(cf in cell_format for cf in ["18650", "2170", "4680"]):
                    continue
                    
                if cell_format not in self.format_to_cells:
                    self.format_to_cells[cell_format] = []
                    self.cell_formats.append(cell_format)
                
                # Gehe bis zum Ordner 'slices' und dann zu den einzelnen Zellen
                slices_dir = os.path.join(self.base_path, format_dir, "slices")
                if os.path.exists(slices_dir):
                    cell_dirs = [d for d in os.listdir(slices_dir) if os.path.isdir(os.path.join(slices_dir, d))]
                    for cell_dir in cell_dirs:
                        # An der Gabelung für 'radial_images' entscheiden
                        radial_dir = os.path.join(slices_dir, cell_dir, "radial_images")
                        if os.path.exists(radial_dir):
                            self.format_to_cells[cell_format].append({
                                'format_dir': format_dir,
                                'radial_dir': radial_dir
                            })

    def __iter__(self):
        return self
        
    def __next__(self):
        if not self.cell_formats:
            raise RuntimeError(f"Keine passenden Daten im Startpfad '{self.base_path}' gefunden. Bitte passen Sie config.BASE_PATH an.")
            
        while True:
            # 1. Uniform Verteilung über Zellformate
            chosen_format = random.choice(self.cell_formats)
            available_cells = self.format_to_cells[chosen_format]
            
            if not available_cells:
                continue
                
            # 2. Uniform Verteilung über die verschiedenen Zellen im Ordner 'slices'
            chosen_cell = random.choice(available_cells)
            radial_dir = chosen_cell['radial_dir']
            
            try:
                images = [img for img in os.listdir(radial_dir) if img.endswith('.png')]
            except OSError:
                continue
                
            if not images:
                continue
                
            # 3. Zufälliges Bild auswählen
            img_name = random.choice(images)
            img_path = os.path.join(radial_dir, img_name)
            
            # Bedingungen extrahieren
            rel_path = os.path.relpath(img_path, self.base_path)
            
            max_height = battery_metadata.get_max_height_from_format(chosen_format)
            abs_depth = battery_metadata.extract_absolute_depth(img_name)
            
            if max_height is None or max_height == 0:
                continue
                
            # Slice depth relativ bestimmen
            rel_depth = abs_depth / max_height
            
            # Überprüfen ob Bild in den gewünschten Grenzen liegt (10% - 90% der maximalen Höhe)
            if self.min_rel_depth <= rel_depth <= self.max_rel_depth:
                try:
                    img = Image.open(img_path)
                    width, height = img.size
                except Exception:
                    continue
                    
                manufacturer = battery_metadata.determine_manufacturer(width, height, rel_path)
                chemistry = battery_metadata.determine_chemistry(manufacturer)
                voxel_size = battery_metadata.determine_voxel_size(chosen_format)
                
                conditions = {
                    "cell_format": chosen_format,
                    "slice_depth_relative": rel_depth,
                    "manufacturer": manufacturer,
                    "chemistry": chemistry,
                    "voxel_size_um": voxel_size
                }
                
                return img, conditions
