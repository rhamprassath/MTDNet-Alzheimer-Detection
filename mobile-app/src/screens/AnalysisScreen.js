import React, { useState, useEffect, useRef } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity, TextInput,
    Animated, StatusBar, SafeAreaView, Dimensions, ScrollView, ActivityIndicator
} from 'react-native';
import axios from 'axios';
import * as DocumentPicker from 'expo-document-picker';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const BACKEND_URL = 'http://192.168.29.19:8000';

// Generates a mock EEG waveform path as an array of {x, y} points
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

// Draw EEG waveform using tiny View bars
function EEGWaveform() {
    const channels = [0, 1, 2, 3];
    const colors = ['#f97316', '#ec4899', '#a855f7', '#3b82f6'];
    const BAR_COUNT = 80;
    const BAR_WIDTH = (SCREEN_WIDTH - 48) / BAR_COUNT;

    return (
        <View style={waveStyles.waveContainer}>
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
                                            opacity: 0.7 + (Math.abs(pt.y) / 30) * 0.3,
                                            marginTop: pt.y > 0 ? 0 : h / 2,
                                        }
                                    ]}
                                />
                            );
                        })}
                    </View>
                );
            })}
        </View>
    );
}

const waveStyles = StyleSheet.create({
    waveContainer: {
        width: '100%', backgroundColor: '#111827',
        borderRadius: 12, padding: 12,
        borderWidth: 1, borderColor: '#1f2937',
        marginBottom: 14,
    },
    channelRow: {
        flexDirection: 'row', alignItems: 'center',
        height: 36, marginVertical: 2, overflow: 'hidden',
    },
    bar: { borderRadius: 1, marginHorizontal: 0.25 },
});

export default function AnalysisScreen({ route, navigation }) {
    const { mode } = route.params;
    const [status, setStatus] = useState('idle'); // idle | running | done
    const [progress, setProgress] = useState(0);
    const [totalSegments, setTotalSegments] = useState(67);
    const [processedSegments, setProcessedSegments] = useState(0);

    // Subject selection for real mode
    const [subjects, setSubjects] = useState([]);
    const [selectedSubject, setSelectedSubject] = useState(null);
    const [loadingSubjects, setLoadingSubjects] = useState(mode === 'real');

    // Upload mode state
    const [uploadedFile, setUploadedFile] = useState(null);

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

        // Animate progress bar
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
                        uploadedFile: mode === 'upload' ? uploadedFile : null
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
            <StatusBar barStyle="light-content" backgroundColor="#0d1117" />
            <View style={styles.statusRow}>
                <Text style={styles.timeText}>9:41</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
                {/* Header */}
                <View style={styles.headerRow}>
                    <TouchableOpacity onPress={() => navigation.goBack()}>
                        <Text style={styles.backBtn}>◀  EEG Analysis</Text>
                    </TouchableOpacity>
                </View>

                {/* Mode badge */}
                <View style={styles.modeBadge}>
                    <Text style={styles.modeBadgeText}>
                        Mode: {mode === 'simulated' ? 'Simulated Stream' : mode === 'upload' ? 'Upload Local EDF/CSV' : `Real Subject — ${selectedSubject || '…'}`}
                    </Text>
                </View>

                {/* EEG Waveform */}
                <EEGWaveform />

                {/* Strategy badge */}
                <View style={styles.strategyBadge}>
                    <Text style={styles.strategyText}>Strategy: Score Averaging</Text>
                </View>

                {/* Subject selector (real mode) */}
                {mode === 'real' && (
                    <View style={styles.subjectSection}>
                        <Text style={styles.subjectLabel}>Select Subject:</Text>
                        {loadingSubjects ? (
                            <ActivityIndicator color="#3b5bdb" />
                        ) : (
                            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
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

                {/* Upload Picker (upload mode) */}
                {mode === 'upload' && (
                    <View style={styles.subjectSection}>
                        <Text style={styles.subjectLabel}>Select EDF/CSV/SET File:</Text>
                        <TouchableOpacity 
                            style={styles.uploadBtn}
                            onPress={async () => {
                                try {
                                    const res = await DocumentPicker.getDocumentAsync({
                                        type: ['*/*'], // EEG formats often don't have standard MIME types
                                    });
                                    if (!res.canceled && res.assets && res.assets.length > 0) {
                                        const file = res.assets[0];
                                        const name = file.name.toLowerCase();
                                        const allowed = ['.edf', '.csv', '.set', '.npz'];
                                        if (!allowed.some(ext => name.endsWith(ext))) {
                                            alert(`Unsupported file format: ${file.name}. Please select an .edf, .csv, .set, or .npz file.`);
                                            return;
                                        }
                                        setUploadedFile(file);
                                    }
                                } catch (e) { console.error('Doc pick error:', e); }
                            }}
                        >
                            <Text style={styles.uploadBtnText}>
                                {uploadedFile ? `Picked: ${uploadedFile.name}` : '📂 Choose File (EDF/CSV)...'}
                            </Text>
                        </TouchableOpacity>
                    </View>
                )}

                {/* Progress bar (shown while running) */}
                {isRunning && (
                    <View style={styles.progressSection}>
                        <View style={styles.progressBarBg}>
                            <Animated.View style={[styles.progressBarFill, { width: progressBarWidth }]} />
                        </View>
                        <Text style={styles.progressText}>
                            Processing {processedSegments} of {totalSegments} segments…
                        </Text>
                    </View>
                )}

                {/* Run MTDNet Inference button */}
                <TouchableOpacity
                    style={[styles.inferenceBtn, isRunning && styles.inferenceBtnDisabled]}
                    onPress={runInference}
                    disabled={isRunning}
                    activeOpacity={0.85}
                >
                    {isRunning ? (
                        <ActivityIndicator color="#fff" />
                    ) : (
                        <Text style={styles.inferenceBtnText}>⚡  Run MTDNet Inference</Text>
                    )}
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#0d1117' },
    statusRow: { paddingHorizontal: 20, paddingTop: 6, alignItems: 'center' },
    timeText: { color: '#d1d5db', fontSize: 13, fontWeight: '500' },
    scroll: { padding: 20, paddingBottom: 40 },

    headerRow: { marginBottom: 16 },
    backBtn: { color: '#93c5fd', fontSize: 16, fontWeight: '700' },

    modeBadge: {
        backgroundColor: '#1f2937', borderRadius: 8,
        paddingHorizontal: 14, paddingVertical: 10,
        marginBottom: 14,
    },
    modeBadgeText: { color: '#94a3b8', fontSize: 14 },

    strategyBadge: {
        backgroundColor: '#1f2937', borderRadius: 8,
        paddingHorizontal: 14, paddingVertical: 10,
        marginBottom: 14, alignSelf: 'stretch',
    },
    strategyText: { color: '#94a3b8', fontSize: 14 },

    subjectSection: { marginBottom: 14 },
    subjectLabel: { color: '#94a3b8', fontSize: 12, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 },
    pill: {
        paddingHorizontal: 16, paddingVertical: 8,
        backgroundColor: '#1f2937', borderRadius: 20,
        marginRight: 8, borderWidth: 1, borderColor: '#374151',
    },
    pillActive: { backgroundColor: 'rgba(59,91,219,0.3)', borderColor: '#3b5bdb' },
    pillText: { color: '#9ca3af', fontSize: 13 },
    pillTextActive: { color: '#fff' },

    uploadBtn: {
        backgroundColor: '#1f2937', padding: 16, borderRadius: 10,
        borderWidth: 1, borderColor: '#3b5bdb', borderStyle: 'dashed',
        alignItems: 'center'
    },
    uploadBtnText: { color: '#93c5fd', fontSize: 14, fontWeight: '600' },

    progressSection: { marginBottom: 14 },
    progressBarBg: {
        height: 28, backgroundColor: '#1f2937', borderRadius: 8,
        overflow: 'hidden', marginBottom: 8,
    },
    progressBarFill: {
        height: '100%', backgroundColor: '#22c55e', borderRadius: 8,
    },
    progressText: { color: '#86efac', fontSize: 13, textAlign: 'center' },

    inferenceBtn: {
        backgroundColor: '#ef4444',
        paddingVertical: 20, borderRadius: 12,
        alignItems: 'center', marginTop: 6,
    },
    inferenceBtnDisabled: { backgroundColor: '#7f1d1d' },
    inferenceBtnText: { color: '#fff', fontSize: 17, fontWeight: '800' }
});
