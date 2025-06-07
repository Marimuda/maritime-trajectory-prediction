Design Document: AIS Data Pipeline for Trajectory Forecasting and Anomaly Detection
Introduction and Data Overview

This design document outlines an offline Python-based pipeline for processing AIS (Automatic Identification System) data from a single-antenna .log file, to prepare data for ship-centric machine learning tasks such as vessel trajectory forecasting and anomaly detection. The input is an AIS log containing millions of AIS messages of various types. Notably, the dataset includes 709,484 Type 1 messages, 81,042 Type 3 messages, 222,453 Type 4 messages, 28,935 Type 5 messages, and a mix of lower-count messages from Types 6–27 (including 40,213 Type 18 and 28,664 Type 24 entries). All information used will be derived from this log (no external data enrichment). The pipeline will ingest and clean these messages, reconstruct per-vessel trajectories, engineer features, filter out noise or errors, and finally output the processed data in convenient formats for ML model development.

This document addresses the key design considerations: selecting relevant message types, integrating Class A vs Class B and static vs dynamic data, reconstructing ordered trajectories per vessel (MMSI), feature engineering for each time-step and trajectory, noise/anomaly filtering strategies, and the output data schema/formats.
1. Relevant AIS Message Types for ML Analysis

AIS messages are categorized by type, each serving a different purpose. For ship movement analysis (trajectory and behavior), the pipeline will focus on dynamic position reports and essential static info while discarding irrelevant messages:

    Dynamic Position Reports (Keep): These messages contain a ship’s GPS position, speed, course, and other kinematic data. We will retain:

        Type 1, 2, 3 – Class A position reports (frequent updates from large vessels)
        documentation.spire.com
        .

        Type 18 – Class B position reports (from smaller vessels, similar data fields to Type 1-3)
        documentation.spire.com
        .

        Type 19 – Extended Class B position reports (if present; includes position plus some static info)
        documentation.spire.com
        .

        Type 27 – Long-range position reports (infrequent, low-resolution position updates for vessels far offshore)
        documentation.spire.com
        .

    Static and Voyage Data (Keep for context): These messages provide vessel identification and characteristics, which are useful for enriching trajectories (e.g. vessel type or dimensions for anomaly context):

        Type 5 – Class A static and voyage data (ship name, call sign, IMO number, ship type, dimensions, destination, etc.)
        documentation.spire.com
        .

        Type 24 – Class B static data reports (usually sent as Part A and Part B; includes name, call sign, and ship dimensions/type for Class B vessels)
        documentation.spire.com
        .

        For Class B, Message 19 (if available) also carries some static fields (like ship type) combined with a position report
        documentation.spire.com
        , and will be processed as a dynamic report augmented with static info.

    Irrelevant or Non-Ship Messages (Discard): Many AIS message types do not contribute to a vessel’s movement track or are not pertinent to ship-centric ML analysis:

        Type 4 – Base station reports (broadcasts of shore station position/time) will be ignored, as they pertain to land-based AIS stations, not vessels
        documentation.spire.com
        .

        Types 6–8, 12–14 – Binary or safety-related messages (addressed/broadcast binary data, text safety messages, acknowledgments) contain no useful trajectory or kinematic information and can be discarded.

        Type 9 – SAR aircraft position reports are not related to vessel trajectories and will be dropped.

        Type 10, 11 – UTC/date inquiry/response, not needed for offline processing since timestamps are available from the log or base messages.

        Types 20, 22, 23 – Base station management messages (data link management, channel management, group assignment) are not relevant to analyzing vessel movement.

        Type 21 – Aids-to-Navigation reports (positions of buoys or navigational aids) will be dropped, since the focus is on ship behavior (these are static objects or infrastructure, not vessels).

        Types 25, 26 – Binary messages (single or multiple slot) have no trajectory data.

By filtering down to dynamic position messages and essential static data, we ensure the pipeline processes only information relevant to ship trajectories and behaviors
documentation.spire.com
. This reduces noise and volume, focusing computational effort on useful data for forecasting and anomaly detection.
2. Data Integration: Class A vs Class B and Static vs Dynamic Data

AIS data must be integrated in a meaningful way to build a coherent dataset per vessel. The pipeline will handle differences between Class A and Class B AIS, and combine static and dynamic information appropriately:

    Class A vs Class B Differences: Class A and Class B transceivers have different reporting frequencies and fields. Class A messages (Types 1–3) come from larger vessels and include fields like navigational status and rate of turn, with high reporting frequency (2–10 seconds when underway at speed, or every 3 minutes when at anchor)
    documentation.spire.com
    . Class B messages (Type 18) come from smaller vessels and generally report less frequently (usually ~30 seconds or more when moving) and at lower power, lacking some fields such as rate of turn or certain status codes. In our pipeline:

        We will not strictly separate the processing of Class A and B streams, but we will mark or handle them as needed. Each incoming message’s type informs how we parse it (e.g., whether to expect ROT or navigational status fields). We will maintain a flag or attribute (e.g., transponder_class = A or B) for each vessel based on the message types seen, so downstream ML models can be aware of class differences (important since Class B data will be sparser and possibly lower accuracy).

        Data structures (e.g., a unified position record schema) will accommodate superset fields (Class A’s extra fields set to null or default when a Class B record is stored). This ensures Class A and B vessel data can reside in the same tables for convenience, while preserving the information of their origin.

    Static vs Dynamic Data Handling: Static (Type 5/24) and dynamic (Type 1/2/3/18/etc.) messages are complementary. The pipeline will maintain two logical sets of data:

        Dynamic Position Table: This will include time-stamped position records (latitude, longitude, speed, course, etc.) for each vessel. Each record will carry the vessel’s MMSI as an identifier.

        Static Information Table: This will store static attributes per vessel (MMSI as key) such as vessel name, call sign, vessel type (cargo, tanker, fishing, etc. as encoded in AIS), length, beam, and any voyage info (destination, ETA) if available. For Class A, these come from Type 5; for Class B, from Type 24 (Parts A and B). The pipeline needs to merge Class B static reports: typically a Class B ship sends Type 24 in two parts (24A with ship name, 24B with dimensions and type). We will combine these parts using MMSI and treat them as one static record. Similarly, if a Class A vessel’s static info is updated (e.g., ETA or destination changes in a new Type 5 message), the static table will be updated (we might keep the latest values or note changes if needed for voyage anomaly detection).

        Joining Static with Dynamic: The static data will be linked to dynamic records by the MMSI identifier. For any given vessel, we can attach static attributes to its position records by MMSI join
        documentation.spire.com
        . In practice, the output might embed static fields into each position record or keep them separate and referential. The design will likely enrich each vessel’s trajectory with relevant static features (like vessel type code), since that can be useful input for ML models (e.g., a cargo ship vs. a passenger ferry may have different motion patterns).

By joining the data, each vessel’s trajectory can be contextualized with its static properties
documentation.spire.com
. For example, the MMSI 123456789 in a position report can be linked to a static record giving the vessel’s name, type, length, etc., enabling features like “turn rate relative to ship size” or filtering anomalies by ship type. The pipeline will ensure static info is readily accessible for each vessel track.

    Handling Message 19 (Extended Class B): If the log contains Type 19 messages, these will be processed as a hybrid of dynamic and static information. Type 19 includes position data (similar to Type 18) and additional static bits (like part of the ship’s name or type)
    documentation.spire.com
    . In our design, we will decode the dynamic fields from Type 19 into the position table and also extract static fields from it to update the static table (especially useful if a Class B vessel did not send a separate Type 24). This ensures no vessel info is missed.

In summary, the pipeline will parse dynamic and static messages in parallel, then integrate them by vessel. Class A and Class B differences will be managed so that the end result is a unified set of vessel trajectories (with consistent fields), each augmented by vessel-specific static data. All integration will use the MMSI as the primary key to join data streams
documentation.spire.com
.
3. Vessel Trajectory Reconstruction per MMSI

Reconstructing clean, ordered trajectories for each vessel (identified by MMSI) is a core function of the pipeline. This involves sorting and cleaning time-series data for each ship, dealing with irregular sampling and any gaps or noise:

    Grouping and Sorting: The pipeline will first group all dynamic position messages by MMSI (each group corresponds to one vessel’s data). Within each group, messages will be sorted by timestamp to form a time-ordered sequence of positions. The log provides timestamps (either via the receiver’s log time or via AIS message fields like Type 4 base station time which can be cross-checked). We will use the reception timestamp for ordering. This yields a chronologically ordered trajectory for each vessel.

    Timestamps and Time Normalization: All timestamps will be converted to a standard format (e.g., UTC datetime or Unix epoch). This ensures consistency when calculating time deltas between messages. If the log’s timestamps are in local time or relative format, they’ll be normalized to absolute UTC times.

    Handling Sampling Rates: AIS data is inherently irregularly sampled – update frequency depends on vessel type, speed, and AIS class. Class A ships broadcast frequently when moving fast (up to every 2–10 seconds) and slower when static (every 3 minutes when anchored)
    documentation.spire.com
    . Class B broadcasts are typically every 30 seconds to a few minutes, depending on movement. This means some trajectories will have dense points, others sparse. The pipeline will preserve the actual timestamps and not assume a fixed interval. For ML applications, we will provide the time delta between consecutive points as a feature (so models can account for varying gap lengths). If needed for specific models, an optional resampling step could be included (e.g., interpolate positions to a fixed time grid), but by default the pipeline will keep original sampling to avoid introducing artificial data.

    Trajectory Segmentation (Missing Data & Gaps): We will detect significant gaps in a vessel’s timeline. If a vessel goes out of reception range or turns off its AIS and then reappears much later, a naive trajectory would have a huge jump. To avoid treating a long gap as continuous motion, the pipeline can split trajectories into segments if the time gap between consecutive messages exceeds a threshold (e.g., > 1 hour or a configurable value). Each segment would represent a continuous voyage or period of coverage. For example, if MMSI 123456789 has no data from January to March, we’d treat data before and after as separate segments for analysis. This prevents unrealistic jumps in the “clean” trajectory. Alternatively, we might mark such gaps so downstream analysis can handle them (e.g., by not drawing straight lines over big time gaps). The output schema could include a segment ID or we simply break output files by segment.

    Eliminating Duplicates and Out-of-order Data: In some cases, duplicate AIS messages might exist in the log (e.g., if the same message was received twice via the same or different channels). We will remove exact duplicate position reports for the same MMSI and timestamp
    coast.noaa.gov
    . Similarly, if any messages are slightly out-of-order (timestamps decreasing), sorting will fix it, and any exact timestamp collisions can be resolved by discarding the later duplicate. This ensures each trajectory is strictly ordered in time.

    Smoothing and Jitter Reduction: AIS positions can exhibit minor jitter (small random jumps around a true position due to GPS error). The pipeline will include basic smoothing options:

        We may apply a median filter or moving average on lat/lon for stationary periods to reduce noise. However, caution is needed not to oversmooth genuine maneuvers. By default, we might not aggressively smooth, but we will clip very small speeds to zero if they likely represent noise (e.g., a vessel at anchor might report 0.2 knots due to GPS drift, which we can treat as 0).

        For trajectory forecasting tasks, a cleaned trajectory (where a stationary ship is clearly stationary) is helpful. The pipeline might set a speed threshold (say below 0.5 knot) to treat as not moving (and possibly hold the position or merge very close points).

        Example: If a vessel’s consecutive points wander within a 50-meter radius while it’s actually docked, we could either leave it (the model can learn it’s effectively stationary by speed ~0) or collapse those into a single point in an optional pre-processing mode for certain analyses like route extraction.

    Coordinate System: All positions will remain in latitude/longitude (WGS84) in the output, but for some feature computations (distance, etc.), the pipeline will internally convert lat/lon differences to meters (using a simple projection or haversine formula). The trajectories are essentially geospatial time-series. If needed, one could project to UTM or local East-North coordinates for a local region analysis, but we will likely keep lat/long for generality and let feature engineering handle distance calculations.

The output of this stage is a set of per-vessel ordered trajectories. Each trajectory is a time-series of points: [(time_1, lat_1, lon_1, sog_1, cog_1, heading_1, ...), (time_2, lat_2, lon_2, ...), ...] for MMSI X, and similarly for each vessel. These form the basis for subsequent feature calculation. The data will be cleaned of obvious inconsistencies so that downstream ML models can train on realistic vessel motion sequences without being confounded by logging artifacts or disordered data.
4. Feature Engineering for Trajectory Data

With clean trajectories per vessel, the pipeline will derive a rich set of features at each time step (and over segments) to support forecasting and anomaly detection tasks. Feature engineering will transform the raw AIS fields into model-friendly inputs that capture vessel kinematics and behavior patterns:

    Basic Kinematic Features (per timestep):

        Speed (SOG) – Speed over ground (knots) as reported, directly used. This is a fundamental feature for both forecasting and identifying abnormal behavior (e.g., unusually high or zero speeds).

        Course (COG) – Course over ground (degrees from North). We will often use the change in course as a feature rather than absolute course, to handle the circular nature of this variable.

        Heading – If available (mostly Class A), the actual heading of the vessel’s bow. The difference between heading and COG can indicate side-slip or currents.

        Latitude & Longitude – While raw coordinates can be used, ML models typically can’t use lat/long directly for prediction without some transformation. We might use relative position or sector as features (e.g., position relative to a route or previous point). At minimum, the pipeline can provide normalized coordinates or compute distances as described below.

    Temporal Features:

        Time Delta – The time difference (in seconds or minutes) between the current message and the previous message. This is crucial when data is irregularly sampled; it allows models to account for varying gaps. It also helps compute derivative features like acceleration.

        Timestamp Components – (Optional) We can derive features like hour of day or day of week from the timestamp if certain behaviors are time-dependent (e.g., port traffic peaks, day vs night patterns). This is more relevant for pattern analysis; forecasting might benefit from periodicity awareness.

    Derived Motion Features (per movement between consecutive points):

        Distance Traveled – Distance between the current position and the previous position (in meters or nautical miles). This is derived via haversine formula on lat/lon and gives a measure of movement magnitude. It correlates with speed but is an independent check (large distance in short time implies high speed or data gap).

        Speed Change (Acceleration/Deceleration) – The change in speed over ground from the last point to the current point (ΔSOG), optionally divided by time delta to give acceleration in knots/hour or m/s². This feature identifies rapid acceleration or deceleration events. Extremely large accelerations might indicate data errors or unusual maneuvers. We will use thresholds based on physical limits to flag unrealistic values
        researchgate.net
        (e.g., a large cargo ship cannot accelerate 10 knots in 30 seconds; such a data point might be noise).

        Course Change (Turn Rate) – The change in course over ground between consecutive points, normalized by time (degrees per minute or per second). This approximates the vessel’s turn rate. AIS message Type 1-3 for Class A actually includes a sensor-reported Rate of Turn (ROT), but for consistency (and for Class B vessels which lack ROT), we compute turn rate from successive COG/heading changes. Large course changes in a short time could indicate sharp maneuvers (or possibly GPS jumps if the value is extreme and inconsistent with speed).

        Course Drift – Difference between the vessel’s reported heading and its course over ground. If heading vs COG differ significantly, the vessel might be sideslipping (due to wind/current) or yawing at anchor. This is more of a situational feature; for anomaly detection, a high drift when moving could be interesting (e.g., a vessel moving sideways).

        Bearing to Previous Point – The initial bearing from the last position to the current position (this ideally should align with COG; if not, it may indicate reporting noise). This can be used to calculate if the vessel is off its expected path segment.

        Turn Radius Estimate – (Optional, advanced) If we have speed and turn rate, we could estimate the radius of turn, which might be characteristic for vessel type or can flag anomalously tight turns for a large ship.

    Behavioral Features over a Window or Segment:

        Average/Max Speed – over a segment or recent window. Useful to characterize the vessel’s typical speed (e.g., a ship normally goes 20 knots; if suddenly 2 knots, maybe it stopped or anomaly).

        Speed Variability – standard deviation of speed in a window, indicating steady vs erratic motion.

        Turn Frequency – how often the vessel changes heading significantly, which might distinguish e.g. fishing behavior (frequent course changes) from transiting.

        Route Deviation – if an expected route is known (not in this scope without external data), but internally we might derive if the vessel’s current movement is linear vs zigzag by looking at successive course changes or comparing to a long-term bearing.

    Spatial Context Features:

        Proximity to Previous Position – Essentially the distance to last point (already noted). We highlight this to catch outliers: if the distance between consecutive points is far larger than expected given the time gap and prior speed, it’s likely a spurious jump. The pipeline will flag or remove such outliers (discussed in noise filtering below). For anomaly detection, the maximum distance between consecutive points can be a feature to detect teleports
        researchgate.net
        .

        Historical Path Features: Though we do not have external route data, we can derive features relative to the vessel’s own recent history. For example, the angle between the incoming and outgoing segment (basically how sharply it turned at this point), or distance from a reference trajectory if we smoothed the past path. These help identify abnormal deviations.

    Static-based Features: Incorporating static data into features can improve ML models:

        Vessel Type Encodings – e.g., cargo, tanker, passenger, fishing (from AIS ship type code
        documentation.spire.com
        ). This can be used as a categorical feature in anomaly detection models (since e.g. fishing vessels might naturally move erratically, whereas passenger ferries follow fixed routes; what’s anomalous for one class might be normal for another).

        Vessel Size – length and width (these can be used to set expectations on maneuverability; e.g., a 300m cargo ship cannot turn or accelerate like a 20m pleasure craft). We might derive a feature like length-to-speed ratio or use size to normalize turn rates (degree per minute per length).

        Draft (if available) – a deeply loaded ship might move more slowly; could be relevant in some contexts (though draft info may not always be present or accurate in AIS).

        Navigational Status – for Class A, nav status (moored, underway, fishing, etc.) is given. This can be directly used or to filter context: for instance, if status is at anchor, we expect speed ~0; any movement could be abnormal or just drift. We will include the nav status code as a feature at each time (possibly one-hot encoded or numeric).

These engineered features will be computed for each successive pair of points, meaning each position record in the output can have not only the raw AIS fields but also *fields like delta_time, distance, speed_delta, turn_rate, etc` derived from it and the previous point. By providing these, the pipeline output spares the modeler from recomputing common trajectory features and ensures consistency in how they’re calculated.

For example, given two consecutive records for a vessel:

    At time T0: position (lat0, lon0), speed = 10.0 kn, course = 90°.

    At time T1: position (lat1, lon1), speed = 12.0 kn, course = 95°.
    The pipeline would output at T1 the derived features such as time_delta = T1-T0, distance = haversine(lat0,lon0, lat1,lon1), speed_delta = +2.0 kn, acceleration = 2.0 kn / time_delta, turn_rate = +5° change in course, etc., and also carry over static info like ship_type = 70 (cargo) for that vessel. These features make it easier to feed the data into anomaly detection algorithms that might, for instance, look for points where acceleration or turn rate exceeds known safe limits
    researchgate.net
    .

The feature engineering step is customizable; additional features can be added as needed for specific ML models. The goal is to capture dynamic behavior patterns in the data representation to enable algorithms to learn vessel motion characteristics and detect when a vessel’s behavior deviates from normal.
5. Noise Filtering and Anomaly Detection in Data

Raw AIS data often contains errors or spurious records that should be filtered out or corrected before using the data for ML. The pipeline will implement a series of rule-based data quality checks to catch and handle these issues:

    Invalid or Absurd Positions: AIS messages sometimes contain default or invalid coordinate values. A known sentinel value is latitude = 91° or longitude = 181°, which indicates “not available” in AIS
    ais.fnb.upc.edu
    . Any message with lat > 90 or lon > 180 (or exactly 91/181) will be discarded
    documentation.spire.com
    , as these are not real positions. Similarly, if the log contains clearly impossible coordinates (e.g., a point on land far from any water when we expect a vessel at sea – though without external coastline data we can’t fully check this), those could be flagged for removal. As an example, a latitude of 91° is not meaningful and is rejected
    cloudeo.group
    .

    Speed Outliers: We will filter out or flag speed values that are beyond physical possibility or highly unlikely for the given vessel:

        If a ship’s SOG is above a reasonable threshold, we consider it an error (e.g., a sudden AIS report of 300 knots is obviously incorrect). Even a value exceeding, say, 50 knots for a large ship is suspect. A common practice is to set an upper speed threshold around 30 knots for anomaly filtering
        sciencedirect.com
        , as very few large vessels exceed 30 knots; anything above that might be noise or a data glitch. The pipeline might use a tiered approach: speeds > 100 knots are dropped outright, speeds 30–100 knots are flagged or could be dropped depending on vessel type (a small craft could go 35 knots, but a cargo ship wouldn’t).

        Zero Speed with High Course Change: A known AIS quirk is when a vessel is stationary (speed ~0) but still reports varying course values. Since course over ground is undefined at zero speed (some AIS units output the last heading or random jitters)
        gpsd.gitlab.io
        sciencedirect.com
        , we will treat any course data when speed is essentially zero as unreliable. If a vessel has SOG = 0 and COG jumping around, we can either freeze the course (e.g., keep the last valid course when moving) or just accept that course is meaningless in that state. For anomaly detection, a situation like “speed=0 but COG=180°” is not a real movement and should not be considered a valid turn. So the pipeline could either nullify COG in such cases or leave it but it’s understood to be noise. This prevents false triggers of “high turn rate” while stationary.

    Position Jumps (Spatial Outliers): If the distance between consecutive points of a vessel is implausibly large given the time gap and speeds, it indicates a likely error or data gap:

        We will compute an expected distance given max speed and time delta. If the actual distance >> expected (say a jump of 50 km when the ship was going 5 knots with a 10 min gap), then that point is probably a teleport. The pipeline can either remove the outlier point or split the trajectory at that point. Another approach is to interpolate a plausible path if the gap is not too large, but since we are not adding external data, by default we’d drop the clearly erroneous point and treat it as missing data.

        We can set a threshold, for example: any single-step jump implying a speed > 100 knots or a distance above, say, 20 nautical miles in one minute would be discarded. Chen et al. (2022) use rules based on maximum distance and vessel maneuverability to detect such anomalies
        researchgate.net
        . Our design will incorporate similar logic – e.g., flag if a point is more than X miles from the previous point where X is far beyond what the vessel could travel in that interval.

    Course/Heading Outliers: Sudden large changes in course can be real (e.g., a fast turn) but if a vessel’s course changes by nearly 180° in one minute at high speed, that might indicate a faulty heading reading or a mismatched MMSI (occasionally, two ships swapping MMSIs in data could cause jumps). We will flag extremely high turn rates as potential data errors. For example, if turn rate exceeds some limit (based on ship type, or a generic threshold like >20° per second for a large ship), we could mark that as an anomaly. Chen et al. refer to maximum angular displacement in a sampling interval as a criterion for anomalies
    researchgate.net
    – the pipeline will use domain knowledge (ships cannot instantaneously pivot beyond certain rates) to filter out impossible heading jumps unless the time delta is large.

    Missing or Inconsistent Static Data: If a vessel’s static message is missing from our log (e.g., we have Class B position reports but never caught its Type 24 static info), the pipeline will still include the trajectory but static fields will be blank/unknown. However, we might flag such cases for completeness. We will not drop an entire vessel track just because static info is missing, since the trajectory itself is valuable. Instead, static features will be null/default in those cases. (If analysis requires static info, the user can filter those out later.)

        Static Data Errors: Static AIS data can also have errors (e.g., incorrect vessel dimensions or MMSI misassignment). As we aren’t using external validation, we will assume the static data as given is mostly correct, but obviously wrong values (like a 0 length or unrealistic ship type code) could be flagged. For example, if a ship type code is not in the known range or MMSI format is wrong (MMSI not 9 digits for ships), we might log a warning or exclude those records
        documentation.spire.com
        (e.g., some coastal station might use a pseudo-MMSI; our filter of discarding non-9-digit MMSIs will naturally drop base stations or other non-vessel stations).

    MMSI Integrity: We will filter messages with invalid MMSIs. Valid vessel MMSIs are nine-digit numbers; base stations use a different format (starting with 00 or 99 and often shorter)
    documentation.spire.com
    . Any message with an MMSI that doesn’t conform to a valid pattern for ships (e.g., too few digits or obviously a test MMSI like 1234567890 with 10 digits) will be discarded
    documentation.spire.com
    to avoid polluting vessel data. This also helps ensure we don’t accidentally treat a base station or aid-to-navigation as a vessel trajectory.

    Duplicate MMSI or Identity Issues: In rare cases, two different ships might erroneously use the same MMSI at different times, or a ship might change its MMSI (should not happen except if a device was moved to another vessel). Our offline pipeline can’t fully resolve identity issues, but we will assume one MMSI = one vessel. If we detect weird scenarios like a jump that could indicate the track actually split into two different vessels using the same MMSI (e.g., the vessel “appears” in two places far apart at the same time), that could be a sign of MMSI reuse or intercept. In such a case, splitting the track is prudent: we would break the trajectory when inconsistency is extreme (the same MMSI showing up 1000 miles apart simultaneously). However, this is an edge case and not explicitly asked in this design, so the default is to treat the data as consistent per MMSI.

    Heuristic Smoothing of Noisy Points: After applying the above filters, we may consider minor smoothing or interpolation for short gaps:

        If a single outlier point was removed, there’s now a gap. We might choose to interpolate a replacement if the gap is small, or just leave the gap. Given “no external data sources” constraint, interpolation would purely be for internal smoothing. We may implement a simple linear interpolation for one-off missing points to avoid breaks in an otherwise smooth trajectory. Alternatively, we leave gaps and let the ML model handle them (some models might be robust to small gaps especially if we include time delta).

        As an optional step for anomaly detection, one might attempt to correct certain anomalies rather than remove them. For example, Chen et al. (2022) used a cubic spline to restore plausible tracks after removing anomalies
        researchgate.net
        . Our pipeline’s scope is primarily detection and filtering; automated restoration is complex and might be future work.

All the above filtering rules ensure that the output trajectories are physically plausible and consistent. Intrinsic anomalies (like impossible values) are removed
cloudeo.group
, and behavioral outliers that likely stem from data errors are filtered so they don’t skew model training. True anomalous vessel behavior (e.g., an illegal deviation) would still remain in the data – the pipeline is careful to remove only data errors, not legitimate unusual maneuvers. For example, a real sharp U-turn by a ship will still be in the data (and an anomaly detection model can catch it), but a nonsensical 1000-knot speed spike will be removed as it provides no useful information and would hinder model training.

In summary, this stage cleans the data by applying domain knowledge constraints: valid ranges for coordinates, speed, acceleration, turn rates, etc., grounded in what ships can actually do
researchgate.net
. It also eliminates duplicates and non-ship messages. The result is a set of vessel trajectories that are ready for use in ML, with minimal risk of garbage-in affecting the algorithms.
6. Output Schema and Data Formats

Finally, the pipeline will output the processed data in structured format(s) suitable for downstream consumption. Since the end goal is to feed into machine learning tasks, we emphasize formats that are easily ingestible in Python/analytics environments and that preserve the per-trajectory grouping. The design will support a few output modes:

    Per-Vessel JSON Lines (JSONL): In this format, each vessel’s trajectory is stored as a single JSON object (or line) in a JSON Lines file. For example, we could have an output file where each line corresponds to one MMSI. This JSON object might look like:

    {
      "mmsi": 123456789,
      "static": {
         "name": "VESSEL NAME",
         "call_sign": "ABCDE",
         "ship_type": 70,
         "length": 200,
         "width": 32,
         "destination": "PORT A",
         "imo": 9876543
      },
      "trajectory": [
         {"timestamp": "2025-05-01T12:00:00Z", "lat": 12.3456, "lon": -98.7654,
          "sog": 10.5, "cog": 85.0, "heading": 84, "nav_status": 0,
          "delta_time": null, "distance": null, "speed_delta": null, "turn_rate": null},
         {"timestamp": "2025-05-01T12:00:30Z", "lat": 12.3460, "lon": -98.7540,
          "sog": 10.7, "cog": 87.0, "heading": 87, "nav_status": 0,
          "delta_time": 30.0, "distance": 1200.0, "speed_delta": 0.2, "turn_rate": 4.0},
         ...
      ]
    }

    Each trajectory array contains the time-ordered points with both raw and derived features. The first point may have delta_time = null since there is no previous point. JSONL per MMSI is human-readable and convenient for debugging or cases where you process vessel by vessel. It also allows nested structure (the trajectory array) which can be useful for certain ML frameworks expecting sequence data in JSON form. However, a downside is file size (a single vessel’s data could be large JSON) and reading it requires parsing JSON for each line.

    Flat Table (CSV/Parquet): We will also provide the output as a flat table of records, where each row is one position record (one timestamp of one vessel), enriched with features and static info. In this table format, we include the MMSI and possibly static attributes duplicated on each row or accessible via a separate lookup. This could be stored as:

        CSV files: A CSV with columns for mmsi, timestamp, lat, lon, sog, cog, heading, nav_status, ship_type, length, ..., delta_time, distance, speed_delta, turn_rate, .... Each row is one message/point. This is a simple and widely compatible format, though the file(s) could be very large given ~1 million records in input. We might partition CSV by date or vessel if needed to manage size.

        Parquet files: Apache Parquet is a columnar binary format that is much more efficient for large data. We will likely output a Parquet file (or partitioned Parquet dataset) for the trajectory data. Parquet offers fast compression and is optimized for analytical queries (e.g., you can easily load only certain columns). Many Python libraries (Pandas, PySpark) can directly ingest Parquet. Given the high volume of AIS records, Parquet is a good choice for performance and storage. For example, a Parquet file with all rows can be quickly filtered by MMSI or time range in downstream code, and it stores columns like latitude and longitude efficiently.

The pipeline might produce multiple Parquet files partitioned by MMSI or by time. One strategy is to partition by MMSI prefix (to avoid too many small files) so that all vessels are distributed among some files, making retrieval of a single vessel's data efficient. Alternatively, partition by date (e.g., one file per day of data) – but since our focus is per vessel, MMSI-based grouping is more logical.

    Static Data Output: Static vessel information (name, type, etc.) could be output as a separate CSV/JSON file mapping MMSI -> static attributes. However, if we embed static info in the main output (each row or in the JSON as above), a separate file might be redundant. We might still output a static reference file for convenience. For example, vessels_static.csv with columns mmsi, name, type_code, type_desc, length, width, ... summarizing each vessel. This can be useful for quickly looking up vessel metadata or doing aggregate analysis by vessel type.

    Schema Definition: Regardless of format, we define the schema of fields clearly. Key fields include:

        MMSI (integer) – vessel identifier
        documentation.spire.com
        .

        Timestamp (ISO datetime or Unix time) – when the message was sent/received
        documentation.spire.com
        .

        Latitude, Longitude (floats) – position coordinates
        documentation.spire.com
        (with a meaningful range check: ±90, ±180).

        Speed (SOG) (float) – in knots
        documentation.spire.com
        .

        Course (COG) (float) – in degrees [0, 360) where 360 or 361 may indicate not available
        documentation.spire.com
        .

        Heading (float or int) – in degrees [0, 359], 511 indicates not available
        documentation.spire.com
        .

        Navigational Status (int) – 0-15 code if available (Class A)
        documentation.spire.com
        .

        Rate of Turn (ROT) (int) – if available (Class A), in 0.1°/s units or “no turn” indicator
        documentation.spire.com
        ; likely we omit if we’re deriving our own turn rate.

        Static fields: ship name (string), call sign, IMO number, vessel type code, length, width, perhaps draft, destination, ETA (if needed)
        documentation.spire.com
        documentation.spire.com
        . Not all static fields are critical for ML; we might include type and size by default.

        Derived features: delta_time (seconds), distance (meters or NM), speed_delta (knots), acceleration (knots per second or m/s²), course_change (degrees), turn_rate (deg/s), etc., as discussed in Section 4.

        We will document units and any null value representations (e.g., None or NaN for missing). Many AIS fields use special values for "not available" (e.g., 102.3 knots for SOG not available, 360° for COG not available)
        documentation.spire.com
        – in our cleaned output we may convert these to nulls or zeros as appropriate after filtering (since we drop invalid positions or set speed=0 when not available, etc.).

An example row in CSV might look like:

mmsi,timestamp,lat,lon,sog,cog,heading,nav_status,ship_type,length,width,delta_time,distance,speed_delta,turn_rate
123456789,2025-05-01T12:00:30Z,12.3460,-98.7540,10.7,87.0,87,0,70,200,32,30.0,1200.0,0.2,4.0

(where nav_status=0 means underway using engine, ship_type=70 means cargo ship, etc.).

    File Organization: Depending on user needs, we can output:

        A single large Parquet file (or dataset directory) containing all trajectory records (with an index or sorted by MMSI, timestamp).

        A CSV file per vessel (not efficient for 100k vessels, so likely not unless focusing on a few vessels).

        Possibly a JSONL file per vessel if that was desired (each file named by MMSI containing that vessel’s JSON lines of data). However, given the phrasing "JSONL per MMSI", it might imply one JSONL file containing all data for each MMSI as separate lines (as described earlier). We will clarify this with the stakeholders, but we lean toward one line per vessel in a common file, or one file per vessel. One line per vessel JSON might become very long if a vessel had thousands of points, so one-file-per-vessel with one line per point (like a CSV but in JSON) is another interpretation. For safety, we can support exporting one file per MMSI (with that file containing the vessel’s track either as JSON lines or CSV lines). This makes it trivial to fetch a single vessel’s history without reading a huge global file.

    Compatibility and Usability: The chosen formats (CSV, Parquet, JSON) are widely used. JSONL is easily read in Python by standard libraries (or line by line), CSV is universal, and Parquet is ideal for large-scale analysis with pandas or Spark. By providing multiple formats, we ensure that data scientists can either do quick looks in CSV/JSON or scale up with Parquet in big data contexts. We note that Spire Maritime, for example, delivers AIS historical data in CSV with separate static and dynamic records that users join by MMSI
    documentation.spire.com
    , and others have moved to Parquet for efficiency in AIS archives. Our pipeline aligns with these best practices by offering a flat table structure that can be optimized (Parquet)
    documentation.spire.com
    .

    Example Output Schema Summary:
    Field	Type	Description
    mmsi	int	Vessel MMSI identifier
    documentation.spire.com
    timestamp	datetime	Timestamp of AIS message (UTC)
    documentation.spire.com
    latitude	float	Latitude in degrees
    documentation.spire.com
    longitude	float	Longitude in degrees
    documentation.spire.com
    sog	float	Speed over ground (knots)
    documentation.spire.com
    cog	float	Course over ground (degrees)
    documentation.spire.com
    heading	float	True heading (degrees)
    documentation.spire.com
    nav_status	int	Navigational status code (if available)
    documentation.spire.com
    ship_type	int	Vessel type code (e.g., 70 = Cargo)
    documentation.spire.com
    length	float	Vessel length (m)
    documentation.spire.com
    width	float	Vessel width (m)
    documentation.spire.com
    ... (other static)		(IMO, call_sign, etc., if needed)
    delta_time	float	Time since previous report (s)
    distance	float	Distance from previous point (m)
    speed_delta	float	Change in speed from previous point (kn)
    acceleration	float	Acceleration (kn/s or m/s²)
    turn_rate	float	Turn rate (deg/s)
    course_change	float	Change in course since previous (deg)
    ... (any other derived)		e.g., curvature, etc., if added

With this schema, the data is self-contained – all needed inputs for trajectory modeling are present per row or per vessel. The use of Parquet or CSV ensures the data can be easily loaded into dataframes for analysis or model training. JSONL provides a more hierarchical view per vessel if needed for certain processing styles.

Finally, we ensure that the output is well-documented and versioned. The design will include metadata (perhaps a README with the schema, units, and any preprocessing applied) so that users of the data (the ML engineers) understand what each field represents, which messages were included or excluded, and any filters applied (e.g., “speeds above X were removed”). This transparency will help in model development and evaluation.

In conclusion, the offline AIS ingestion pipeline will produce a comprehensive, cleaned, and enriched dataset of vessel trajectories, ready for machine learning applications. The key design choices – focusing on relevant AIS messages, merging static info, ordering per MMSI, deriving features like speed changes and turn rates, filtering out noisy data points
researchgate.net
cloudeo.group
, and outputting in accessible formats – all serve to maximize the usefulness of the data for trajectory forecasting and anomaly detection in the maritime domain. The structured output will allow data scientists to jump straight into analysis and modeling, confident in the quality and completeness of the underlying data.

Sources:

    AIS message type definitions and categories
    documentation.spire.com
    documentation.spire.com
    documentation.spire.com

    AIS data integration via MMSI (linking static and dynamic records)
    documentation.spire.com
    documentation.spire.com

    AIS data quality and filtering heuristics (invalid coordinates, MMSI, etc.)
    documentation.spire.com
    cloudeo.group

    Maneuver-based anomaly detection features (speed/acceleration, distance, angle thresholds)
