Of course. That is an excellent strategy. Separating the work into two distinct papers—one introducing the dataset and another detailing the detection method—is a common and highly effective academic practice. It allows each paper to have a clear, focused contribution and can lead to two separate publications.

Here is a revised structure for a paper that focuses exclusively on introducing the **dataset itself**. The methodology is now framed as a brief example use case to demonstrate the dataset's value, not as the central contribution.

-----

### **A Channel-Aware AIS Dataset for Maritime Security Research**

**Jóhannus Kristmundsson, *Member, IEEE*, Jákup Svøðstein**

\<br\>

***Abstract***—*The security of the Automatic Identification System (AIS) is a growing concern, yet research into advanced spoofing detection has been hampered by a lack of public datasets that include physical-layer information. Most available AIS data is limited to kinematic and message content, omitting crucial signal characteristics that can reveal transmitter authenticity. This paper introduces a new, publicly available dataset designed to fill this gap. [cite\_start]Collected over a 10-day period in May 2025, the dataset contains 2,012,183 AIS messages from 999 unique vessels[cite: 35, 36]. [cite\_start]Uniquely, each message is enriched with physical-layer metadata captured by a software-defined radio (SDR) receiver, including received signal power and frequency offset (PPM)[cite: 18, 31]. We provide a detailed description of the data collection methodology, the dataset's structure, and a comprehensive characterization of its features. To demonstrate its utility, we present a brief use case involving a graph-based anomaly detection model that successfully identifies outliers based on these channel-aware features. This dataset provides a valuable resource for developing and validating next-generation maritime security systems, particularly those focused on spoofing detection and radiometric fingerprinting.*

***Index Terms***—*Dataset, AIS, Maritime Security, Physical-Layer Security, Spoofing Detection, Software-Defined Radio (SDR).*

-----

#### **I. BACKGROUND AND MOTIVATION**

[cite\_start]The Automatic Identification System (AIS) is a critical component of global maritime infrastructure, supporting collision avoidance and vessel traffic services[cite: 3, 4]. [cite\_start]Despite its importance, the system's lack of authentication mechanisms makes it vulnerable to spoofing attacks, where malicious actors transmit false information to create "ghost ships" or impersonate legitimate vessels[cite: 6, 7, 8].

[cite\_start]While several methods have been proposed to detect spoofing based on kinematic inconsistencies[cite: 9], they can be defeated by sophisticated attacks. [cite\_start]A more robust approach involves analyzing physical-layer characteristics of the radio signal, which are inherently difficult to forge[cite: 11, 19]. However, progress in this area has been limited by a significant gap: the scarcity of public AIS datasets that include this physical-layer metadata. This paper introduces a new dataset to address this need and enable more advanced security research.

#### **II. DATA COLLECTION METHODOLOGY**

**A. Receiver Setup**
AIS messages were collected using a stationary, shore-based receiver. The hardware consisted of:

  * [cite\_start]An RTL-SDR v3 USB dongle[cite: 26].
  * [cite\_start]A vertically polarized VHF antenna mounted 60 meters above sea level to ensure a good line-of-sight over the monitored area[cite: 27].

[cite\_start]The system was configured to monitor both AIS channels (161.975 MHz and 162.025 MHz)[cite: 27]. [cite\_start]To ensure the integrity of power measurements, automatic gain control (AGC) was disabled, and a fixed RF gain of 20 dB was maintained throughout the collection period[cite: 28]. [cite\_start]The open-source `AIS-catcher` software was used for real-time GMSK demodulation and decoding, providing access to both the message content and the physical-layer metadata[cite: 29, 30].

**B. Data Acquisition and Structure**
[cite\_start]The data acquisition campaign was conducted continuously from May 8 to May 17, 2025[cite: 35]. [cite\_start]A total of 2,012,183 valid AIS messages were collected from 999 distinct vessels[cite: 35, 36]. [cite\_start]A coverage map of all received positions is shown in Fig. 1 in the original draft[cite: 38, 49].

Each record in the dataset corresponds to a single decoded AIS message and includes the following fields:

  * [cite\_start]**Vessel Identifier:** Maritime Mobile Service Identity (MMSI)[cite: 31].
  * [cite\_start]**Temporal Information:** UTC timestamp[cite: 33].
  * [cite\_start]**Kinematic Information:** Latitude, longitude, Speed Over Ground (SOG), Course Over Ground (COG), and heading[cite: 31, 34].
  * [cite\_start]**Physical-Layer Metadata:** Received signal power (in dB), frequency offset (in PPM), and the AIS channel (A or B)[cite: 31, 33].

#### **III. DATASET CHARACTERIZATION AND VALIDATION**

**A. Signal Power Analysis**
[cite\_start]The received signal power across all messages ranged from -39.4 dB to +7.4 dB[cite: 50]. [cite\_start]A strong relationship between signal power and distance from the antenna was observed, closely following a free-space path loss (FSPL) model in the 10-40 km range[cite: 52]. [cite\_start]At distances less than 5 km, signal power flattens due to clipping in the SDR's front-end, a common artifact in high-SNR conditions[cite: 39, 40].

**B. Frequency Offset Analysis**
[cite\_start]The frequency offset, expressed in parts-per-million (PPM), provides a coarse radiometric fingerprint of the transmitting radio hardware[cite: 45]. [cite\_start]Analysis of the dataset revealed that PPM values were highly consistent for each vessel over time, suggesting that oscillator error is the dominant factor[cite: 46]. [cite\_start]While a weak Doppler shift was also detected, the stability of the baseline offset makes PPM a valuable feature for device-specific identification and spoofing detection[cite: 47, 54, 56]. [cite\_start]The PPM values in the dataset exhibit noticeable quantization, with a dominant step size of approximately 0.289 PPM, a characteristic of the decoding software's frequency estimation algorithm[cite: 58].

**C. Data Quality and Preprocessing**
To ensure the dataset is ready for research applications, a comprehensive preprocessing pipeline was applied:

1.  [cite\_start]**Trajectory Segmentation:** Position reports from each vessel were grouped into trajectories, with a new segment created after any temporal gap exceeding 30 minutes[cite: 70].
2.  [cite\_start]**Filtering:** Short trajectory segments (\<6 points) were discarded[cite: 71]. [cite\_start]Kinematic outlier filtering was performed to remove data points violating plausible speed, acceleration, or turn rate thresholds[cite: 76, 77].
3.  [cite\_start]**Data Integration:** Dynamic position reports were merged with static vessel information (e.g., vessel type, dimensions) from AIS message types 5 and 24 to create enriched trajectory profiles[cite: 78, 79].

#### **IV. POTENTIAL RESEARCH APPLICATIONS**

This dataset is designed to support a wide range of research in maritime systems and security.

  * [cite\_start]**AIS Spoofing Detection:** The primary application is the development and testing of algorithms that use physical-layer features (power, PPM) to detect spoofed messages[cite: 19].
  * [cite\_start]**Radiometric Fingerprinting:** The stability of PPM values allows for research into identifying and tracking specific AIS transceivers based on their unique hardware characteristics[cite: 12, 47].
  * **Signal Propagation Modeling:** The data can be used to study VHF signal propagation in a maritime environment.
  * **Enhanced Vessel Tracking:** The physical-layer data provides an additional dimension for improving the accuracy and integrity of vessel tracking algorithms.

[cite\_start]To demonstrate the dataset's utility, we trained a baseline graph autoencoder model[cite: 22]. [cite\_start]The model successfully used the channel features to identify anomalies, including injected synthetic spoofed data and real-world outliers with airborne characteristics, confirming the data's value for security research[cite: 24, 25, 61].

#### **V. DATA AVAILABILITY**

The complete dataset, including raw logs, preprocessed trajectory files, and accompanying Python scripts for parsing and analysis, is publicly available. It is hosted at [**Name of Repository, e.g., IEEE DataPort, Zenodo, or GitHub**] under a [**Name of License, e.g., Creative Commons CC BY 4.0**] license. The dataset can be accessed via the following persistent identifier: [**DOI or Link to Repository**].

-----

**REFERENCES**
[1] A. Harati-Mokhtari, A. Wall, P. Brooks, and J. Wang. "Automatic identification system (ais): Data reliability and human error implications," *Journal of Navigation*, vol. 60, no. 3, p. [cite\_start]373-389, 2007[cite: 116].
...
*[Full reference list from the original document to follow]*
