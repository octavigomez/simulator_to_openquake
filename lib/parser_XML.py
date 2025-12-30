"""

This code has two functions: one to parse ruptures within a catalogue and
extract the necessary parameters and store them into an array, another to convert to XML in the format
required by OpenQuake for characteristic sources.

Authors: Octavi Gómez-Novell
Institution: CN Instituto Geológico y Minero de España (IGME-CSIC)

Location: Madrid, Spain
Last update: October 2025

License: CC-BY-NC 4.0

"""

from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom


def parse_ruptures(rupt_cat):
    ruptures = []
    for row in rupt_cat:
        if len(row) < 4:
            continue
        rupture_id = int(float(row[0]))
        magnitude = row[1]
        occur_rate = row[2]
        rake = row[3]
        coords = row[4].split()
        # Group coordinates into (lon, lat, z) tuples
        pos_list = []
        if len(coords) > 4:
            for i in range(0, len(coords), 3):
                if i + 2 < len(coords):
                    lon, lat, z = coords[i], coords[i + 1], coords[i + 2]
                    pos_list.extend([str(lon), str(lat), str(z)])

            ruptures.append({
                'id': str(rupture_id),
                'magnitude': str(magnitude),
                'occur_rate': str(occur_rate),
                'rake': str(rake),
                'pos_list': pos_list
            })
    return ruptures


def create_xml(ruptures, output_file):
    # Create root element
    nrml = Element('nrml', {
        'xmlns': 'http://openquake.org/xmlns/nrml/0.5',
        'xmlns:gml': 'http://www.opengis.net/gml'
    })

    # Create source model
    source_model = SubElement(nrml, 'sourceModel', {'name': 'Converted Rupture Model'})

    # Create source group
    source_group = SubElement(source_model, 'sourceGroup', {
        'name': 'rupture group',
        'tectonicRegion': 'Active Shallow Crust'
    })

    # Add each rupture as a characteristicFaultSource
    for rupture in ruptures:
        source = SubElement(source_group, 'characteristicFaultSource', {
            'id': str(rupture['id']),
            'name': f"rupture {rupture['id']}"
        })

        # Add MFD
        arbitrary_mfd = SubElement(source, 'arbitraryMFD')
        SubElement(arbitrary_mfd, 'occurRates').text = rupture['occur_rate']
        SubElement(arbitrary_mfd, 'magnitudes').text = rupture['magnitude']

        # Add rake (using -90 as in example)
        SubElement(source, 'rake').text = rupture['rake']

        # Add surface
        surface = SubElement(source, 'surface')
        gridded_surface = SubElement(surface, 'griddedSurface')
        pos_list = SubElement(gridded_surface, 'gml:posList')
        pos_list.text = ' '.join(rupture['pos_list'])

    # Convert to pretty XML
    xml_str = minidom.parseString(tostring(nrml)).toprettyxml(indent='    ')

    # Write to file
    with open(output_file, 'w') as f:
        f.write(xml_str)