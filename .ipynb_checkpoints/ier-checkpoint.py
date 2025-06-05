from collections import Counter
import math
# Sample documents
d1 = "Chandrayaan is a series of Indian lunar space probes. Chandrayaan-1, the first lunar space probe of the Indian Space Research Organisation (ISRO), discovered water on the Moon. It mapped the Moon in infrared, visible, and X-ray light from lunar orbit and used reflected radiation to prospect for various elements, minerals, and ice. It operated in 2008–09. Chandrayaan-2, launched in 2019, was designed to be ISRO’s first lunar lander. Chandrayaan-3, ISRO’s first lunar lander, touched down in the Moon’s south polar region in 2023."
d2 = "A Polar Satellite Launch Vehicle launched the 590-kg (1,300-pound) Chandrayaan-1 on October 22, 2008, from the Satish Dhawan Space Centre on Sriharikota Island, Andhra Pradesh state. The probe was then boosted into an elliptical polar orbit around the Moon, reaching a closest distance of 504 km (312 miles) to the lunar surface and a farthest distance of 7,502 km (4,651 miles). After checkout, it descended to a 100-km (60-mile) orbit. On November 14, 2008, Chandrayaan-1 launched a small craft, the Moon Impact Probe (MIP), designed to test systems for future landings and study the thin lunar atmosphere before crashing on the Moon’s surface. MIP impacted near the south pole and, before crashing, discovered small amounts of water in the Moon’s atmosphere."
d3 = "The U.S. National Aeronautics and Space Administration (NASA) contributed two instruments, the Moon Mineralogy Mapper (M3) and the Miniature Synthetic Aperture Radar (Mini-SAR), which sought ice at the poles. M3 studied the lunar surface in wavelengths from the visible to the infrared to isolate signatures of different minerals on the surface. It found small amounts of water and hydroxyl radicals on the Moon’s surface. M3 also discovered in a crater near the Moon’s equator evidence for water coming from beneath the surface. Mini-SAR broadcast polarized radio waves at the north and south polar regions. Changes in the polarization of the echo measured the dielectric constant and porosity, which are related to the presence of water ice. The European Space Agency (ESA) had two other experiments, an infrared spectrometer and a solar wind monitor. The Bulgarian Aerospace Agency provided a radiation monitor."
d4 = "The principal instruments from ISRO—the Terrain Mapping Camera, the HyperSpectral Imager, and the Lunar Laser Ranging Instrument—produced images of the lunar surface with high spectral and spatial resolution, including stereo images with a 5-metre (16-foot) resolution and global topographic maps with a resolution of 10 metres (33 feet). The Chandrayaan Imaging X-ray Spectrometer, developed by ISRO and ESA, was designed to detect magnesium, aluminum, silicon, calcium, titanium, and iron by the X-rays they emit when exposed to solar flares. This was done in part with the Solar X-Ray Monitor, which measured incoming solar radiation."
d5 = "Chandrayaan-1 operations were originally planned to last two years, but the mission ended on August 28, 2009, when radio contact was lost with the spacecraft. Chandrayaan-2 launched on July 22, 2019, from Sriharikota on a Geosynchronous Satellite Launch Vehicle Mark III. The spacecraft consisted of an orbiter, a lander, and a rover. The orbiter circles the Moon in a polar orbit at a height of 100 km (62 miles) and has a planned mission lifetime of seven and a half years. The mission’s Vikram lander (named after ISRO founder Vikram Sarabhai) was planned to land on September 7. Vikram carried the small (27-kg [60- pound]) Pragyan (Sanskrit: “Wisdom”) rover. Both Vikram and Pragyan were designed to operate for 1 lunar day (14 Earth days). However, just before Vikram was to touch down on the Moon, contact was lost at an altitude of 2 km (1.2 miles)."
d6 = "Chandrayaan-3 launched from Sriharikota on July 14, 2023. The spacecraft consists of a Vikram lander and a Pragyan rover. The Vikram lander touched down on the Moon on August 23. It became the first spacecraft to land in the Moon’s south polar region where water ice could be found under the surface. The landing site was the farthest south that any lunar probe had touched down, and India was the fourth country to have landed a spacecraft on the Moon—after the United States, Russia, and China."

# List of documents
documents = [d1, d2, d3, d4, d5, d6]

# Initialize the 2D map
term_frequency_map = {}

# Iterate through each document
for i, document in enumerate(documents, start=1):
    # Tokenize the document into words
    # print(i)
    words = document.lower().split()  # Convert to lowercase for case-insensitivity

    # Count the frequency of each term
    term_frequency = Counter(words)

    # Store the term frequencies in the 2D map
    for term, count in term_frequency.items():
        if term not in term_frequency_map:
            term_frequency_map[term] = {}
        term_frequency_map[term][i] = count

# # Display the 2D map
# for term, term_count_map in term_frequency_map.items():
#     print(f"{term}: {term_count_map}")
# Iterate through term frequency map and apply transformation
# Iterate through term frequency map and apply transformation
print("Term frequency matrix")
for term, term_count_map in term_frequency_map.items():
    print(term,term_count_map,sep='-')
    mx=0;
    for i,count in term_count_map.items():
        mx=max(mx,count)
        # # Print with a custom separator
        # print(term, i, count, sep="-")
    for i,count in term_count_map.items():
        term_frequency_map[term][i] = 0.5+((0.5*count)/(mx*count))




# Initialize the dictionary to store the count of documents for each term
term_document_count = {}

# Iterate through the term frequency map
for term, term_count_map in term_frequency_map.items():
    # Count the number of documents the term is present in
    num_documents = len(term_count_map)

    # Store the count in the dictionary
    term_document_count[term] = num_documents

for key,val in term_document_count.items():
    term_document_count[key]=math.log2(1+6/val)

for term,count_map in term_frequency_map.items():
    for i,count in count_map.items():
        term_frequency_map[term][i]=term_frequency_map[term][i]*term_document_count[term]

# Get the unique list of document identifiers
document_identifiers = sorted(set(document for term_count_map in term_frequency_map.values() for document in term_count_map))

# Initialize the 2D matrix with zeros
term_document_matrix = [[0 for _ in document_identifiers] for _ in term_frequency_map]

# Fill in the matrix with term frequencies
for i, (term, term_count_map) in enumerate(term_frequency_map.items()):
    for j, document in enumerate(document_identifiers):
        term_document_matrix[i][j] = term_count_map.get(document, 0)

# Display the 2D matrix
print("Term-IDF Matrix:")
for row in term_document_matrix:
    print(row)


import numpy as np
# Convert the term_document_matrix to a NumPy array
term_document_matrix_np = np.array(term_document_matrix)

# Multiply the matrix with its transpose
term_term_corr = term_document_matrix_np @ term_document_matrix_np.T

# Display the result
print("Term Term Correlation Matrix:")
print(term_term_corr)






