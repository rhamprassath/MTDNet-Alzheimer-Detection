import React, { useState, useEffect, useRef } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity, TextInput,
    Animated, StatusBar, SafeAreaView, Dimensions, ScrollView, ActivityIndicator
} from 'react-native';
import axios from 'axios';
import * as DocumentPicker from 'expo-document-picker';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const BACKEND_URL = 'http://10.26.143.148:8000';

// Generates a mock EEG waveform path
function generateWavePoints(count = 120, channelIndex = 0) {
    const points = [];
    for (let i = 0; i < count; i++) {
        const t = (i / count) * Math.PI * 8;
        const alpha = Math.sin(t * (1.2 + channelIndex * 0.15)) * 18;
        const theta = Math.sin(t * 2.5 + channelIndex) * 10;
        const noise = (Math.random() - 0.5) * 8;
        points.push({ x: i, y: alpha + theta + noise });
    }
    return points;
}

// Scifi Animated Waveform
function EEGWaveform({ isProcessing }) {
    const channels = [0, 1, 2, 3];
    const colors = ['#f97316', '#ec4899', '#a855f7', '#3b82f6'];
    const BAR_COUNT = 80;
    const BAR_WIDTH = (SCREEN_WIDTH - 64) / BAR_COUNT;

    return (
        <View style={[waveStyles.waveContainer, isProcessing && waveStyles.waveProcessing]}>
            {channels.map((ch, ci) => {
                const pts = generateWavePoints(BAR_COUNT, ch);
                return (
                    <View key={ci} style={waveStyles.channelRow}>
                        {pts.map((pt, i) => {
                            const h = Math.abs(pt.y) + 4;
                            return (
                                <View
                                    key={i}
                                    style={[
                                        waveStyles.bar,
                                        {
                                            height: h,
                                            width: BAR_WIDTH - 0.5,
                                            backgroundColor: colors[ci],
                                            opacity: isProcessing ? 0.9 : 0.4 + (Math.abs(pt.y) / 40),
                                            marginTop: pt.y > 0 ? 0 : h / 2,
                                        }
                                    ]}
                                />
                            );
                        })}
                    </View>
                );
            })}
            {isProcessing && (
                <View style={waveStyles.scanLine} />
            )}
        </View>
    );
}

const waveStyles = StyleSheet.create({
    waveContainer: {
        width: '100%', backgroundColor: 'rgba(5, 11, 20, 0.8)',
        borderRadius: 16, padding: 16,
        borderWidth: 1, borderColor: '#1E293B',
        marginBottom: 20, position: 'relative', overflow: 'hidden'
    },
    waveProcessing: {
        borderColor: '#3B82F6',
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
    },
    channelRow: {
        flexDirection: 'row', alignItems: 'center',
        height: 36, marginVertical: 2, overflow: 'hidden',
    },
    bar: { borderRadius: 1, marginHorizontal: 0.25 },
    scanLine: {
        position: 'absolute', top: 0, bottom: 0, left: '50%', width: 2,
        backgroundColor: '#60A5FA', shadowColor: '#60A5FA', shadowOpacity: 1, shadowRadius: 10,
    }
});

export default function AnalysisScreen({ route, navigation }) {
    const { mode, patientId } = route.params || { mode: 'upload', patientId: null };
    const [status, setStatus] = useState('idle');
    const [progress, setProgress] = useState(0);
    const [totalSegments, setTotalSegments] = useState(67);
    const [processedSegments, setProcessedSegments] = useState(0);

    const [subjects, setSubjects] = useState([]);
    const [selectedSubject, setSelectedSubject] = useState(null);
    const [loadingSubjects, setLoadingSubjects] = useState(mode === 'real');

    const [uploadedFile, setUploadedFile] = useState(null);
    const [patientName, setPatientName] = useState('');
    const [patientAge, setPatientAge] = useState('');
    const [patientDetails, setPatientDetails] = useState('');

    const progressAnim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        if (mode === 'real') {
            axios.get(`${BACKEND_URL}/subjects`, { timeout: 5000 })
                .then(res => {
                    const subs = res.data.subjects || [];
                    setSubjects(subs);
                    if (subs.length > 0) setSelectedSubject(subs[0]);
                })
                .catch(() => { })
                .finally(() => setLoadingSubjects(false));
        }
    }, [mode]);

    const runInference = () => {
        if (mode === 'upload' && !uploadedFile) {
            alert("Please pick an .edf or .set file first!");
            return;
        }

        setStatus('running');
        setProgress(0);
        setProcessedSegments(0);

        let seg = 0;
        const total = totalSegments;
        const interval = setInterval(() => {
            seg += Math.ceil(total / 20);
            if (seg >= total) seg = total;
            setProcessedSegments(seg);
            const pct = seg / total;
            setProgress(pct);
            Animated.timing(progressAnim, {
                toValue: pct, duration: 300, useNativeDriver: false,
            }).start();
            if (seg >= total) {
                clearInterval(interval);
                setStatus('done');
                setTimeout(() => {
                    navigation.replace('Results', { 
                        mode, 
                        subjectId: selectedSubject, 
                        uploadedFile: mode === 'upload' ? uploadedFile : null,
                        patientId, patientName, patientAge, patientDetails
                    });
                }, 600);
            }
        }, 300);
    };

    const progressBarWidth = progressAnim.interpolate({
        inputRange: [0, 1],
        outputRange: ['0%', '100%'],
    });

    const isRunning = status === 'running';

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#050B14" />
            <LinearGradient colors={['#0A0F1D', '#050B14']} style={StyleSheet.absoluteFillObject} />
            
            <View style={styles.headerRow}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={{flexDirection: 'row', alignItems: 'center'}}>
                    <Feather name="chevron-left" size={24} color="#60A5FA" />
                    <Text style={styles.backBtn}>Back</Text>
                </TouchableOpacity>
            </View>

            <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
                <View style={styles.modeBadge}>
                    <Text style={styles.modeBadgeText}>
                        {mode === 'upload' ? 'Upload Mode' : mode === 'real' ? 'Real Database' : 'Simulation Mode'}
                    </Text>
                    <Text style={styles.brandSubtitle}>Initializing Pipeline</Text>
                </View>

                {/* Scifi Waveform */}
                <EEGWaveform isProcessing={isRunning} />

                {/* Database Selection (real mode) */}
                {mode === 'real' && (
                    <View style={styles.subjectSection}>
                        <Text style={styles.sectionLabel}>Available Subjects</Text>
                        {loadingSubjects ? (
                            <ActivityIndicator size="small" color="#60A5FA" style={{ marginVertical: 10 }} />
                        ) : (
                            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ flexDirection: 'row' }}>
                                {subjects.map(sub => (
                                    <TouchableOpacity 
                                        key={sub}
                                        style={[styles.pill, selectedSubject === sub && styles.pillActive]}
                                        onPress={() => setSelectedSubject(sub)}
                                    >
                                        <Text style={[styles.pillText, selectedSubject === sub && styles.pillTextActive]}>
                                            {sub}
                                        </Text>
                                    </TouchableOpacity>
                                ))}
                            </ScrollView>
                        )}
                    </View>
                )}

                {/* Patient Information & File Upload (upload mode) */}
                {mode === 'upload' && (
                    <View style={styles.subjectSection}>
                        <Text style={styles.sectionLabel}>Patient Metadata</Text>
                        <TextInput 
                            style={styles.inputField} 
                            placeholderTextColor="#475569" 
                            placeholder="Patient Full Name" 
                            value={patientName} 
                            onChangeText={setPatientName} 
                        />
                        <View style={{flexDirection: 'row', justifyContent: 'space-between', gap: 12}}>
                            <TextInput 
                                style={[styles.inputField, {flex: 1}]} 
                                placeholderTextColor="#475569" 
                                placeholder="Age" 
                                keyboardType="numeric"
                                value={patientAge} 
                                onChangeText={setPatientAge} 
                            />
                            <TextInput 
                                style={[styles.inputField, {flex: 2}]} 
                                placeholderTextColor="#475569" 
                                placeholder="Clinical Notes..." 
                                value={patientDetails} 
                                onChangeText={setPatientDetails} 
                            />
                        </View>

                        <Text style={[styles.sectionLabel, { marginTop: 16 }]}>EDF / CSV Signal Target</Text>
                        <TouchableOpacity 
                            style={[styles.uploadDropzone, (!patientName || !patientAge) && { opacity: 0.4 }]}
                            activeOpacity={0.7}
                            onPress={async () => {
                                if (!patientName || !patientAge) {
                                    alert("Please supply Patient Name and Age prior to targeting data.");
                                    return;
                                }
                                try {
                                    const res = await DocumentPicker.getDocumentAsync({ type: ['*/*'] });
                                    if (!res.canceled && res.assets && res.assets.length > 0) {
                                        const file = res.assets[0];
                                        const allowed = ['.edf', '.csv', '.set', '.npz'];
                                        if (!allowed.some(ext => file.name.toLowerCase().endsWith(ext))) {
                                            alert(`Unsupported format: ${file.name}`);
                                            return;
                                        }
                                        setUploadedFile(file);
                                    }
                                } catch (e) { console.error(e); }
                            }}
                        >
                            <Feather name="upload-cloud" size={32} color={uploadedFile ? '#4ADE80' : '#64748B'} marginBottom={10} />
                            <Text style={[styles.uploadDropzoneText, uploadedFile && { color: '#4ADE80' }]}>
                                {uploadedFile ? `Targeted: ${uploadedFile.name}` : 'Tap to Target File'}
                            </Text>
                        </TouchableOpacity>
                    </View>
                )}

                {/* Process Execution Section */}
                <View style={[styles.actionSection, isRunning && { opacity: 0.8 }]}>
                    {isRunning && (
                        <View style={styles.progressContainer}>
                            <View style={styles.progressBarWrapper}>
                                <Animated.View style={[styles.progressBarFill, { width: progressBarWidth }]} />
                            </View>
                            <Text style={styles.progressLabel}>
                                Synthesizing {processedSegments} / {totalSegments} epochs...
                            </Text>
                        </View>
                    )}

                    <TouchableOpacity
                        style={styles.deployBtn}
                        activeOpacity={0.8}
                        onPress={runInference}
                        disabled={isRunning || status === 'done'}
                    >
                        <LinearGradient
                            colors={['#2563EB', '#4F46E5']}
                            start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }}
                            style={styles.deployBtnGradient}
                        >
                            {isRunning ? (
                                <ActivityIndicator size="small" color="#fff" />
                            ) : (
                                <Text style={styles.deployBtnText}>Execute Neural Analysis</Text>
                            )}
                        </LinearGradient>
                    </TouchableOpacity>
                </View>

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#050B14' },
    scroll: { padding: 24, paddingBottom: 60 },
    headerRow: { 
        paddingHorizontal: 20, paddingTop: 10, paddingBottom: 10,
        flexDirection: 'row', alignItems: 'center' 
    },
    backBtn: { color: '#60A5FA', fontSize: 16, fontWeight: '700', marginLeft: 4 },

    modeBadge: { marginBottom: 24, alignItems: 'flex-start' },
    modeBadgeText: { color: '#F8FAFC', fontSize: 26, fontWeight: '900', letterSpacing: 0.5 },
    brandSubtitle: { fontSize: 14, color: '#94A3B8', fontWeight: '500', marginTop: 4, textTransform: 'uppercase', letterSpacing: 1 },

    sectionLabel: { color: '#94A3B8', fontSize: 13, marginBottom: 12, textTransform: 'uppercase', letterSpacing: 1, fontWeight: '700' },
    
    subjectSection: { marginBottom: 30 },
    pill: {
        paddingHorizontal: 20, paddingVertical: 12,
        backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 24,
        marginRight: 10, borderWidth: 1, borderColor: '#1E293B',
    },
    pillActive: { backgroundColor: 'rgba(37,99,235,0.2)', borderColor: '#3B82F6' },
    pillText: { color: '#94A3B8', fontSize: 15, fontWeight: '600' },
    pillTextActive: { color: '#E2E8F0' },

    inputField: {
        backgroundColor: 'rgba(255,255,255,0.03)',
        color: '#F8FAFC',
        borderRadius: 14,
        padding: 18,
        marginBottom: 12,
        fontSize: 15,
        borderWidth: 1,
        borderColor: '#1E293B'
    },

    uploadDropzone: {
        backgroundColor: 'rgba(255,255,255,0.01)', padding: 30, borderRadius: 16,
        borderWidth: 1.5, borderColor: '#334155', borderStyle: 'dashed',
        alignItems: 'center', justifyContent: 'center'
    },
    uploadDropzoneText: { color: '#94A3B8', fontSize: 15, fontWeight: '600' },

    actionSection: { marginTop: 10 },
    progressContainer: { marginBottom: 20 },
    progressBarWrapper: {
        height: 8, backgroundColor: '#1E293B', borderRadius: 4,
        overflow: 'hidden', marginBottom: 10,
    },
    progressBarFill: {
        height: '100%', backgroundColor: '#60A5FA', borderRadius: 4,
        shadowColor: '#60A5FA', shadowOpacity: 0.8, shadowRadius: 10, elevation: 4
    },
    progressLabel: { color: '#94A3B8', fontSize: 13, fontWeight: '600', alignSelf: 'center' },

    deployBtn: { width: '100%', borderRadius: 16, overflow: 'hidden' },
    deployBtnGradient: { paddingVertical: 20, alignItems: 'center', justifyContent: 'center' },
    deployBtnText: { color: '#FFF', fontSize: 16, fontWeight: '800', letterSpacing: 0.5 },
});
