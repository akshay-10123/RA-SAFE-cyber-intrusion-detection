# 04. Attack Categories & Cyber-Physical Threat Analysis

This document provides a detailed examination of the 5 flight classes represented in the UAV Cyber-Physical Intrusion Detection System.

---

## 1. Class 0: Benign (Normal Flight)

### Definition
Benign operation represents legitimate, uncompromised UAV flight under nominal pilot commands and standard environmental conditions.

### Attacker Actions
None. Spectrum is free of adversarial transmission.

### UAV Behavior
The quadcopter responds smoothly to pilot stick inputs, maintains stable attitude control, hovers within expected velocity bounds ($\le 2\text{ m/s}$ in hover), and keeps a stable Wi-Fi connection.

### Cyber Indicators
- Monotonically increasing 802.11 sequence numbers (`wlan.seq` advances by +1 per frame).
- Stable signal-to-noise ratio (`wlan_radio.SNR (db)` typically between 20 and 35 dB).
- Dominance of data frames (`wlan.fc.type = 2`) and expected beacon intervals.
- Normal frame durations (`wlan.duration`).

### Physical Indicators
- Accelerometer readings hovering around nominal gravity vector: $a_{gz} \approx 9.8\text{ m/s}^2$, $a_{gx}, a_{gy} \approx 0\text{ m/s}^2$.
- Euler angles (pitch, roll, yaw) fluctuate smoothly within nominal deadbands ($\pm 5^\circ$).
- Smooth velocity derivatives consistent with physics.

### Dataset Representation
- Rows $0 - 13,716$ (13,717 samples, 25.04% of dataset).
- Physical variables follow nominal zero-mean distributions.

---

## 2. Class 1: De-authentication Denial of Service (DoS)

### Definition
A wireless cyberattack that exploits IEEE 802.11 management frames to forcibly sever the communication link between the UAV and the Ground Control Station (GCS).

### Attack Mechanism
In standard 802.11 Wi-Fi without 802.11w Protected Management Frames (PMF), management frames are unauthenticated and unencrypted. An attacker within RF range sniffs the MAC address of the drone and pilot, crafts spoofed 802.11 Type 0, Subtype 12 (`Deauthentication`) frames purporting to come from the access point, and broadcasts them at high frequency:

```
[Attacker Laptop / Alfa Wi-Fi Adapter]
        │
        ├── Spoofed Deauth Frame: Type=0, Subtype=12 ──► [UAV Drone]
        │   (Source: Pilot MAC, Dest: Drone MAC)
        │
        └── Spoofed Deauth Frame: Type=0, Subtype=12 ──► [Pilot Smartphone / GCS]
            (Source: Drone MAC, Dest: Pilot MAC)
```

### UAV Consequence
The UAV Wi-Fi stack immediately tears down the association. The pilot loses live video and control telemetry. The drone switches to autonomous failsafe mode (hovering in place, initiating auto-land, or triggering Return-to-Home).

### Cyber Indicators
- Dominance of management frames: `wlan.fc.type = 0`, `wlan.fc.subtype = 12`.
- Burst of frames with `data.len = 0` (management frames carry no upper-layer payload).
- Disrupted sequence numbers (`wlan.seq` anomalies).
- Elevated `wlan.duration` values.

### Physical Indicators
- Abrupt cessation of pilot-commanded maneuvers; drone transitions into hover or descent.
- Sensor variance shifts as flight controller engages auto-stabilization failsafe.

### Real-World Example
An unauthorized intruder uses a handheld ESP8266 "deauther" or Kali Linux laptop with `aireplay-ng --deauth 100 -a <BSSID> -c <UAV_MAC> wlan0` to disconnect a commercial drone securing an industrial perimeter, blinding the security operator.

---

## 3. Class 2: Replay Attack

### Definition
A cryptographic and transmission attack where an eavesdropper records legitimate telemetry and control packets sent earlier during flight and re-broadcasts them at a later time.

### Attack Mechanism
The adversary does not need to decrypt or reverse-engineer the packet payloads. By capturing valid flight command packets (e.g., "fly forward at 5 m/s") during takeoff, the attacker replays those exact bytes over the wireless channel while the drone is near an obstacle:

```
Step 1 (Eavesdropping): Pilot sends valid "Pitch Forward" packet -> Attacker logs packet.
Step 2 (Replay Injection): Drone approaches tree -> Attacker broadcasts captured "Pitch Forward" packet.
Step 3 (Execution): Drone accepts stale valid packet -> Drone accelerates into tree.
```

### UAV Consequence
The flight controller receives stale setpoints. If the drone is moving toward a structure, replaying old acceleration setpoints overrides pilot braking commands, causing kinetic crashes.

### Cyber Indicators
- Stale or non-monotonic sequence control counters (`wlan.seq` jumps backwards).
- Identical payload lengths (`data.len`) and frame flags repeating unnaturally.
- Timing jitter discrepancies (`wlan.duration`).

### Physical Indicators
- Aircraft kinematics conflict with real-time situational demands.
- Sustained velocity vectors when attitude feedback would dictate braking.

### Real-World Example
In a drone delivery corridor, an adversary captures radio packets during a high-speed transit phase and replays them when the drone descends near the delivery drop zone, causing it to crash into the ground.

---

## 4. Class 3: Evil Twin Attack

### Definition
A rogue access point attack where an adversary sets up a counterfeit Wi-Fi network broadcasting the exact SSID, channel, and spoofed MAC address of the legitimate UAV access point, transmitting at a higher transmission power ($P_{\text{tx}}$).

### Attack Mechanism
1. The attacker deploys a rogue access point mimicking the UAV Ground Station SSID (e.g., `"TELLO-FE34A2"`).
2. The attacker transmits a brief de-authentication burst to force the drone to disconnect from the pilot.
3. Because Wi-Fi clients automatically re-associate with the strongest signal matching the SSID, the drone connects to the attacker's rogue AP.

```
[Legitimate Ground Station] (RSSI: -65 dBm) 
         ▲
         │ (De-authenticated)
      [UAV Drone]
         │ (Re-associates with stronger signal)
         ▼
[Adversary Evil Twin AP] (RSSI: -30 dBm, High Gain Antenna)
```

### UAV Consequence
The adversary attains a Man-in-the-Middle (MitM) position, enabling full packet inspection, command modification, GPS waypoint injection, or deliberate hijacking of the video stream.

### Cyber Indicators
- Sudden step shift in RF signal quality (`radiotap.signal_quality`) and `wlan_radio.SNR (db)` without physical position change.
- Alterations in AP capabilities, datarates (`wlan_radio.datarate`), or beacon interval headers.
- Potential mismatch between `wlan.bssid` and observed RF parameters.

### Physical Indicators
- Drone remains physically stationary or enters unexpected loitering patterns while communication handshakes occur with the counterfeit station.

### Real-World Example
An attacker at a public sporting event deploys a high-power rogue access point to hijack a media broadcast drone, rerouting its flight path to a remote landing site for physical theft.

---

## 5. Class 4: False Data Injection (FDI)

### Definition
A physical and sensor-level attack where an adversary directly tampers with sensor measurements, state estimation feedback, or control setpoints.

### Attack Mechanism
Unlike DoS or Evil Twin attacks which disrupt the network pipe, FDI targets data integrity:
- **RF Sensor Spoofing**: Transmitting counterfeit GPS signals (GPS spoofing) or acoustic resonance signals against MEMS gyroscopes.
- **Payload Manipulation**: Modifying telemetry packets in transit (via MitM) to report that the drone is at an altitude of 100 meters when it is actually at 10 meters:

```
[Ground Sensor / Altitude ToF] ──► Reads: 10 meters
                                        │
                        (Adversary modifies payload)
                                        │
[Flight Controller Estimation] ──► Believes: 100 meters ──► Motors throttle down ──► Drone crashes
```

### UAV Consequence
The flight controller's state estimator (Extended Kalman Filter) computes erroneous state updates. The autopilot attempts to correct for non-existent deviations, causing the physical airframe to flip, stall, or dive into the ground.

### Cyber Indicators
- Cyber packets appear structurally normal: valid 802.11 headers, valid CRC (`wlan.fcs.status = 0`), normal sequence counters (`wlan.seq`), and legitimate data frame types (`wlan.fc.type = 2`).
- **Network-only IDSs almost completely miss pure FDI attacks.**

### Physical Indicators
- **Severe physical anomalies**: Non-aerodynamic step jumps in Euler angles (`pitch`, `roll`, `yaw`).
- Inconsistent accelerations ($a_{gx}, a_{gy}, a_{gz}$) violating aerodynamic drag equations.
- Inconsistency between integrated velocity $\int a \, dt$ and measured ground speed.
- Discontinuous jumps in Time-of-Flight altitude (`tof`).

### Real-World Example
An attacker uses a Software-Defined Radio (HackRF One) to transmit GPS spoofing signals to an autonomous surveillance drone, causing the autopilot to register false position drift and fly off-course into restricted airspace.

---

## 6. Threat Comparative Matrix

| Attack Type | Primary Domain | Target Asset | Packet Validity | Physical Impact | Key Detection Modality |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Benign** | None | None | Normal | Normal Flight | Both nominal |
| **DoS (Deauth)** | Cyber | Availability | Malicious Management Frames | Transition to Failsafe | **Cyber Branch** |
| **Replay** | Cyber / Link | Integrity | Stale Legitimate Frames | Unintended Latent Motion | **Cyber Branch (Sequence/Timing)** |
| **Evil Twin** | Cyber / Physical Layer | Authenticity | Rogue AP Framing | Delayed Pilot Response | **Cyber Branch (Signal/BSSID)** |
| **FDI** | Physical / Telemetry | Integrity | **Structurally Valid Frames** | **Catastrophic Attitude Divergence** | **Physical Branch (Kinematics)** |

### Scientific Takeaway for Defense
- **DoS, Replay, and Evil Twin** primarily perturb the **Cyber Domain** (802.11 protocols, sequence numbers, RF metrics).
- **FDI** primarily perturbs the **Physical Domain** (attitude, kinematics, sensor plausibility) while remaining stealthy in network headers.
- **Therefore, a dual-branch cyber-physical architecture is strictly required to protect both domains.**
