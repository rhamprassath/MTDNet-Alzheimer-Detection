import AsyncStorage from '@react-native-async-storage/async-storage';
import 'react-native-get-random-values';
import { v4 as uuidv4 } from 'uuid';

const PATIENTS_KEY = '@mtdnet_patients';

export const getPatients = async () => {
    try {
        const jsonValue = await AsyncStorage.getItem(PATIENTS_KEY);
        return jsonValue != null ? JSON.parse(jsonValue) : [];
    } catch (e) {
        console.error("Failed to load patients", e);
        return [];
    }
};

export const savePatient = async (patient) => {
    try {
        const patients = await getPatients();
        const existingIndex = patients.findIndex(p => p.id === patient.id);
        
        if (existingIndex >= 0) {
            patients[existingIndex] = patient;
        } else {
            patients.push(patient);
        }
        
        await AsyncStorage.setItem(PATIENTS_KEY, JSON.stringify(patients));
        return true;
    } catch (e) {
        console.error("Failed to save patient", e);
        return false;
    }
};

export const createPatient = async (name, age, gender, notes) => {
    const newPatient = {
        id: uuidv4(),
        name,
        age,
        gender,
        notes,
        createdAt: new Date().toISOString(),
        history: []
    };
    await savePatient(newPatient);
    return newPatient;
};

export const addTestToPatient = async (patientId, testResult) => {
    try {
        const patients = await getPatients();
        const existingIndex = patients.findIndex(p => p.id === patientId);
        
        if (existingIndex >= 0) {
            const resultWithDate = {
                ...testResult,
                date: new Date().toISOString()
            };
            patients[existingIndex].history.unshift(resultWithDate); // Add newest to front
            await AsyncStorage.setItem(PATIENTS_KEY, JSON.stringify(patients));
            return true;
        }
        return false;
    } catch (e) {
        console.error("Failed to save test result", e);
        return false;
    }
};

export const deletePatient = async (patientId) => {
    try {
         const patients = await getPatients();
         const updated = patients.filter(p => p.id !== patientId);
         await AsyncStorage.setItem(PATIENTS_KEY, JSON.stringify(updated));
         return true;
    } catch (e) {
         return false;
    }
};
