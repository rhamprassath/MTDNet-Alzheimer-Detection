import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView, StatusBar, ScrollView, Animated } from 'react-native';
import { getPatients, deletePatient } from '../utils/storage';

export default function PatientProfileScreen({ route, navigation }) {
    const { patientId } = route.params;
    const [patient, setPatient] = useState(null);

    useEffect(() => {
        const load = async () => {
            const patients = await getPatients();
            const found = patients.find(p => p.id === patientId);
            setPatient(found);
        };
        const unsubscribe = navigation.addListener('focus', () => {
             load();
        });
        load();
        return unsubscribe;
    }, [patientId, navigation]);

    if (!patient) return null;

    const handleDelete = async () => {
        await deletePatient(patientId);
        navigation.navigate('Home');
    };

    const getDiagnosisColor = (diagnosis) => {
        if (!diagnosis) return '#64748b';
        const d = diagnosis.toLowerCase();
        if (d.includes('healthy')) return '#22c55e'; // Green
        if (d.includes('mild') || d.includes('mci')) return '#facc15'; // Yellow
        return '#ef4444'; // Red
    };

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" />
            <View style={styles.header}>
                <TouchableOpacity onPress={() => navigation.goBack()}>
                    <Text style={styles.backBtn}>◀ Back</Text>
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Patient Profile</Text>
                <View style={{ width: 50 }} />
            </View>

            <ScrollView contentContainerStyle={styles.content}>
                <View style={styles.profileCard}>
                    <View style={styles.avatar}>
                        <Text style={styles.avatarText}>{patient.name.charAt(0)}</Text>
                    </View>
                    <Text style={styles.name}>{patient.name}</Text>
                    <Text style={styles.details}>{patient.age} yrs • {patient.gender}</Text>
                    {patient.notes ? <Text style={styles.notes}>"{patient.notes}"</Text> : null}
                </View>

                <TouchableOpacity 
                    style={styles.newTestBtn} 
                    onPress={() => navigation.navigate('Analysis', { mode: 'upload', patientId: patient.id })}
                >
                    <Text style={styles.newTestBtnText}>+ Run New Analysis</Text>
                </TouchableOpacity>

                <Text style={styles.sectionTitle}>Progression History</Text>
                
                {patient.history && patient.history.length > 0 ? (
                    patient.history.map((test, index) => {
                         const color = getDiagnosisColor(test.diagnosis);
                         return (
                            <View key={index} style={[styles.historyCard, { borderLeftColor: color }]}>
                                <View style={styles.historyHeader}>
                                    <Text style={styles.historyDate}>
                                        {new Date(test.date).toLocaleDateString()}
                                    </Text>
                                    <Text style={[styles.historyConf, { color }]}>{test.confidence}</Text>
                                </View>
                                <Text style={[styles.historyDiag, { color }]}>{test.diagnosis}</Text>
                            </View>
                         );
                    })
                ) : (
                    <Text style={styles.emptyText}>No test history for this patient.</Text>
                )}

                <TouchableOpacity style={styles.deleteBtn} onPress={handleDelete}>
                    <Text style={styles.deleteBtnText}>Delete Patient Profile</Text>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#0d1117' },
    header: {
        flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
        paddingHorizontal: 20, paddingVertical: 16, borderBottomWidth: 1, borderBottomColor: '#1f2937'
    },
    backBtn: { color: '#93c5fd', fontSize: 16, fontWeight: '600' },
    headerTitle: { color: '#f8fafc', fontSize: 18, fontWeight: '700' },
    content: { padding: 20 },
    profileCard: {
        backgroundColor: '#111827', borderRadius: 12, padding: 24,
        alignItems: 'center', marginBottom: 20, borderWidth: 1, borderColor: '#1f2937'
    },
    avatar: {
        width: 80, height: 80, borderRadius: 40, backgroundColor: '#3b5bdb',
        justifyContent: 'center', alignItems: 'center', marginBottom: 12
    },
    avatarText: { fontSize: 36, color: '#fff', fontWeight: '800' },
    name: { fontSize: 24, color: '#f8fafc', fontWeight: '800', marginBottom: 4 },
    details: { fontSize: 16, color: '#94a3b8', marginBottom: 12 },
    notes: { fontStyle: 'italic', color: '#64748b', textAlign: 'center' },
    newTestBtn: {
        backgroundColor: '#10b981', paddingVertical: 16, borderRadius: 10,
        alignItems: 'center', marginBottom: 30
    },
    newTestBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },
    sectionTitle: { fontSize: 18, color: '#e2e8f0', fontWeight: '700', marginBottom: 16 },
    historyCard: {
        backgroundColor: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12,
        borderLeftWidth: 4
    },
    historyHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
    historyDate: { color: '#94a3b8', fontSize: 14 },
    historyConf: { fontWeight: '700', fontSize: 14 },
    historyDiag: { fontSize: 16, fontWeight: '700' },
    emptyText: { color: '#64748b', fontStyle: 'italic', marginBottom: 30 },
    deleteBtn: { marginTop: 40, alignItems: 'center', paddingVertical: 12 },
    deleteBtnText: { color: '#ef4444', fontSize: 14, fontWeight: '600' }
});
