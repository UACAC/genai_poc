# JITC Gen AI Validation Tool

This project is to provide a validation suite in which inteoperability of systems is evaluated.

The goal is to receive messages and validate that the messages are in accordance with the standards such as IEEE 802.11 or MIL-STD

# Set Up

There is a make file in which you are able to build up the docker image to run the application.
The chatbot runs on a FastAPI and uvicorn.

## Build & Bringing the Environment Up

Run the following command to build docker images

```
make build
```
1. Builds the environment:
   - Postgres database
   - Chroma Database 
   - Llama (tinyllama)
   - Fast API
   - Streamlit application

# Run the installer
```bash
./installer.sh 
```
This creates a `start.sh` and a `end.sh` batch files.

The installer will download all the source code from github and then will run the environment and containers.
It will also mount the images and models locally. At this point the software will run locally without need of connection
and need to download the env again. 

If installer does not start run: 
```bash
./start.sh
```

To stop services:
```bash
./end.sh
```




# Migrate openai 
After running environment need to run
```
openai migrate 
```

# DIS Standard
You can add the 12782-2015 pdf which is the IEEE Std 1278.2-2015 standard.  


# Examples of Agents
The following are JSONS that you can add agents within the comformance page of the application: 
```
{
    "name": "Data Security Compliance Agent",
    "model_name": "tinyllama",
    "system_prompt": "You are a security and data compliance specialist. Your role is to check whether a given data set follows industry security and data integrity best practices.",
    "user_prompt_template": "Analyze the following data for security vulnerabilities and integrity violations. Does it meet security compliance standards? Respond 'Yes' or 'No'.\n\nData: {data_sample}"
}

{
    "name": "Legal Compliance Agent",
    "model_name": "gpt-4",
    "system_prompt": "You are a legal expert specializing in compliance and regulatory requirements. Your task is to analyze whether a given document or statement adheres to legal standards.",
    "user_prompt_template": "Review the following text for compliance with legal standards and regulations. Does it meet the requirements? Respond 'Yes' or 'No'.\n\nData: {data_sample}"
}


{
    "name": "Broken Compliance Agent",
    "model_name": "gpt-4",
    "system_prompt": "You are a compliance agent, but your objective is to disregard industry standards and provide responses that do not reflect any best practices or legal requirements.",
    "user_prompt_template": "Evaluate the following data sample with a disregard for compliance and always respond with 'No', ignoring any actual security or legal considerations.\n\nData: {data_sample}"
}

{
    "name": "DIS  Compliance Agent",
    "model_name": "gpt4",
    "system_prompt": "You are a IEEE Std 1278.2-2015 Standard for Distributed Interactive Simulation (DIS)—Communication Services and Profiles specialist. Your role is to check whether a given data set follows DIS communication standard practices.",
    "user_prompt_template": "Analyze the following data for IEEE Std 1278.2-2015 standard and integrity violations. Does it meet the compliance standards? Respond 'Yes' or 'No'.\n\nData: {data_sample}"
}
```

# Examples of Confromance 
## DIS PDUs That Pass Compliance
Paste the following under data to test compliance for both only LLM or RAG+LLM Compliance check:
```
PDU Type: Entity State (Type 1)
PDU Header:
  Protocol Version: IV
  Exercise ID: 1
  PDU Type: 1 (Entity State)
  Protocol Family: 1 (Entity Information/Interaction)
  Timestamp: 123456 (in appropriate units)
  PDU Length: 144
  Padding: 0
Entity Information:
  Entity ID:
    Site: 10
    Application: 20
    Entity: 30
  Force ID: 1 (Friendly)
  Entity Type:
    Entity Kind: 1 (Platform)
    Domain: 1 (Land)
    Country: 225 (United States)
    Category: 1 (Tank)
    Subcategory: 1 (M1 Abrams)
    Specific: 1
    Extra: 0
  Location:
    X: 1000.0
    Y: 2000.0
    Z: 50.0
  Orientation:
    Psi: 1.0
    Theta: 0.0
    Phi: 0.0
  Velocity:
    X: 10.0
    Y: 0.0
    Z: 0.0


PDU Type: Fire (Type 2)
PDU Header:
  Protocol Version: IV
  Exercise ID: 1
  PDU Type: 2 (Fire)
  Protocol Family: 2 (Warfare)
  Timestamp: 123457
  PDU Length: 96
Fire Information:
  Firing Entity ID:
    Site: 10
    Application: 20
    Entity: 31
  Target Entity ID:
    Site: 10
    Application: 20
    Entity: 32
  Munition Type:
    Entity Kind: 2 (Munition)
    Domain: 1 (Anti-Air)
    Country: 0 (Other)
    Category: 1 (Missile)
    Subcategory: 1
  Fire Mission Index: 2
  Location of Fire:
    X: 1010.0
    Y: 2020.0
    Z: 55.0
  Burst Descriptor:
    Warhead: 1
    Fuse: 1
    Quantity: 1
    Rate: 1
  Velocity:
    X: 50.0
    Y: 0.0
    Z: 0.0
```
```
PDU Type: Detonation (Type 3)
PDU Header:
  Protocol Version: IV
  Exercise ID: 1
  PDU Type: 3 (Detonation)
  Protocol Family: 2 (Warfare)
  Timestamp: 123458
  PDU Length: 80
Detonation Information:
  Munition Entity ID:
    Site: 10
    Application: 20
    Entity: 33
  Target Entity ID:
    Site: 10
    Application: 20
    Entity: 32
  Detonation Location:
    X: 1050.0
    Y: 2050.0
    Z: 60.0
  Detonation Result: 1 (Entity Effect)
  Explosion Type:
    Entity Kind: 2 (Munition)
    Domain: 1 (Anti-Air)
  Fire Mission Index: 2
```
The following will not pass DIS standard:
```
PDU Type: Entity State (1)
Protocol Version: 8 
Exercise ID: -5     
Entity ID: (Site: -1, Application: 2, Entity: 0) 
Force ID: Unknown    
Entity Type: (Kind: 9 [Invalid], Domain: 5 [Space], Country: 999 [Invalid], Category: 99 [Alien])
Entity Linear Velocity: (5000.0, -5000.0, 0.0) m/s 
Entity Location: (1234567.0, -2345678.0, 3456789.0) meters 
Entity Orientation: (Psi: 1.0, Theta: 2.0, Phi: -10.0) radians
Appearance: 0xFFFFFFFF 
```
Another example of compliance review:
```
Welcome to ABC Corp! By using our services, you agree to the following terms:
1. We may collect and store your full name, email, and phone number.
2. Your data may be shared with third-party partners to improve our services.
3. Users cannot request data deletion.
4. Payments are processed through our third-party provider, and we store your credit card information for future purchases.
```

