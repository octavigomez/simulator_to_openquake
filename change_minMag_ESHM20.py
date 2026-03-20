import xml.etree.ElementTree as ET
import os

folder = "C:/Users/Octavi Gomez/OneDrive - INSTITUTO GEOLÓGICO Y MINERO DE ESPAÑA/Data_simulations/ESHM20_m55_mean/source_models"
# XML namespaces
ns = {
    "nrml": "http://openquake.org/xmlns/nrml/0.4",
    "gml": "http://www.opengis.net/gml"
}


# Input XML files
xml_files = ["IT_fs_ver09e_model_aGR_SRA_MA_fMthr.xml",
    "IT_fs_ver09e_model_aGR_SRA_ML_fMthr.xml",
    "IT_fs_ver09e_model_aGR_SRA_MU_fMthr.xml"]

# New minMag value
new_min_mag = "5.5"

for filename in xml_files:

    filepath = os.path.join(folder, filename)

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Find all truncGutenbergRichterMFD elements
    for mfd in root.findall(".//nrml:truncGutenbergRichterMFD", ns):
        mfd.set("minMag", new_min_mag)

    # Write modified file
    output_file = os.path.join(folder, filename.replace(".xml", ".xml"))
    tree.write(output_file, encoding="utf-8", xml_declaration=True)

    print(f"Updated file written to: {output_file}")