import React, { useEffect, useState, useRef } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity,
    Animated, StatusBar, SafeAreaView, ScrollView, ActivityIndicator
} from 'react-native';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const BACKEND_URL = 'http://192.168.29.19:8000';

// Small horizontal bar indicator for EEG feature
function IndicatorBar({ value = 0.6, color = '#22c55e' }) {
    return (
        <View style={barStyles.track}>
            <View style={[barStyles.fill, { width: `${Math.min(value * 100, 100)}%`, backgroundColor: color }]} />
        </View>
    );
}
const barStyles = StyleSheet.create({
    track: { height: 8, backgroundColor: '#1f2937', borderRadius: 4, flex: 1, overflow: 'hidden' },
    fill: { height: '100%', borderRadius: 4 },
});

export default function ResultsScreen({ route, navigation }) {
    const { mode, subjectId, uploadedFile } = route.params;
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(true);
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const scaleAnim = useRef(new Animated.Value(0.92)).current;

    useEffect(() => {
        const fetchResult = async () => {
            try {
                let res;
                if (mode === 'simulated') {
                    res = await axios.post(`${BACKEND_URL}/evaluate/simulated`, {
                        simulation_len_seconds: 10, sampling_rate: 128, strategy: 'average'
                    }, { timeout: 15000 });
                } else if (mode === 'upload' && uploadedFile) {
                    // Create FormData
                    const formData = new FormData();
                    // Web vs Native handling
                    if (uploadedFile.file) {
                        formData.append('file', uploadedFile.file);
                    } else {
                        formData.append('file', {
                            uri: uploadedFile.uri,
                            name: uploadedFile.name,
                            type: uploadedFile.mimeType || 'application/octet-stream'
                        });
                    }
                    // sampling_rate is now auto-sensed by backend
                    res = await axios.post(`${BACKEND_URL}/evaluate/upload`, formData, {
                        headers: { 'Accept': 'application/json' },
                        timeout: 60000 
                    });
                } else {
                    res = await axios.post(`${BACKEND_URL}/evaluate/real`, {
                        subject_id: subjectId || 'sub-001', strategy: 'average'
                    }, { timeout: 15000 });
                }
                setResult(res.data);
                saveToHistory(res.data);
            } catch (err) {
                console.error("API Error:", err);
                let errorMsg = 'API Unreachable — showing demo data';
                if (err.response && err.response.data && err.response.data.detail) {
                    errorMsg = typeof err.response.data.detail === 'string' 
                        ? err.response.data.detail 
                        : JSON.stringify(err.response.data.detail);
                } else if (err.message) {
                    errorMsg = err.message;
                }
                setResult({
                    diagnosis: 'Analysis Failed',
                    confidence: '0.00%',
                    segments_analyzed: 0,
                    strategy: '-',
                    evaluation_time_seconds: 0.0,
                    error: errorMsg,
                });
            } finally {
                setLoading(false);
            }
        };

        const saveToHistory = async (data) => {
            try {
                const jsonValue = await AsyncStorage.getItem('@mtdnet_history');
                const history = jsonValue != null ? JSON.parse(jsonValue) : [];
                
                let subj = 'Simulation';
                if (mode === 'real') subj = subjectId;
                if (mode === 'upload') subj = uploadedFile.name;

                const newEntry = {
                    date: new Date().toISOString(),
                    subject: subj,
                    diagnosis: data.diagnosis,
                    confidence: data.confidence,
                };
                
                history.push(newEntry);
                await AsyncStorage.setItem('@mtdnet_history', JSON.stringify(history));
            } catch (e) {
                console.error("Failed to save history:", e);
            }
        };

        fetchResult();
    }, [mode, subjectId, uploadedFile]);

    useEffect(() => {
        if (!loading) {
            Animated.parallel([
                Animated.timing(fadeAnim, { toValue: 1, duration: 700, useNativeDriver: true }),
                Animated.spring(scaleAnim, { toValue: 1, friction: 5, tension: 50, useNativeDriver: true }),
            ]).start();
        }
    }, [loading]);

    if (loading) {
        return (
            <SafeAreaView style={styles.safe}>
                <StatusBar barStyle="light-content" backgroundColor="#0d1117" />
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#3b5bdb" />
                    <Text style={styles.loadingText}>Synthesizing Results…</Text>
                </View>
            </SafeAreaView>
        );
    }

    // Determine colors based on diagnosis
    const diag = result.diagnosis || '';
    const isHealthy = diag.toLowerCase().includes('healthy');
    const isAD = diag.toLowerCase().includes('alzheimer') || diag.toLowerCase().includes('ad');
    const diagColor = isHealthy ? '#22c55e' : isAD ? '#ef4444' : '#f59e0b';
    const diagBg = isHealthy ? '#052e16' : isAD ? '#450a0a' : '#431407';
    const diagBorder = isHealthy ? '#16a34a' : isAD ? '#b91c1c' : '#b45309';

    // Parse confidence number
    const confStr = typeof result.confidence === 'string' ? result.confidence : `${(result.confidence * 100).toFixed(2)}%`;

    // Mock EEG Indicators (derived from diagnosis)
    const alphaPower = isHealthy ? 0.78 : 0.32;
    const thetaRatio = isHealthy ? 0.45 : 0.81;

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#0d1117" />
            <View style={styles.statusRow}>
                <Text style={styles.timeText}>9:41</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
                {/* Header */}
                <View style={styles.headerRow}>
                    <Text style={styles.headerIcon}>🔬</Text>
                    <Text style={styles.headerTitle}>Diagnostic Results</Text>
                </View>

                {/* Diagnosis card */}
                <Animated.View
                    style={[
                        styles.diagCard,
                        { backgroundColor: diagBg, borderColor: diagBorder },
                        { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }
                    ]}
                >
                    <Text style={styles.diagLabel}>DIAGNOSIS</Text>
                    <Text style={[styles.diagValue, { color: diagColor }]}>{diag}</Text>
                </Animated.View>

                {/* Confidence + Segments row */}
                <View style={styles.statsRow}>
                    <View style={[styles.statCard, { borderColor: '#1e40af' }]}>
                        <Text style={styles.statLabel}>PERCENTAGE</Text>
                        <Text style={[styles.statValue, { color: '#60a5fa' }]}>{confStr}</Text>
                    </View>
                    <View style={[styles.statCard, { borderColor: '#92400e' }]}>
                        <Text style={styles.statLabel}>SEGMENTS</Text>
                        <Text style={[styles.statValue, { color: '#fbbf24' }]}>
                            {result.segments_analyzed} segs
                        </Text>
                    </View>
                </View>

                {/* Strategy + Time info */}
                <View style={styles.infoRow}>
                    <Text style={styles.infoRowText}>
                        Strategy: {result.strategy === 'average' ? 'Score Averaging' : 'Consensus'}
                        {'  |  '}
                        Time: {result.evaluation_time_seconds || '0.31'}s
                    </Text>
                </View>

                {/* Error notice if demo data */}
                {result.error && (
                    <View style={styles.errorBanner}>
                        <Text style={styles.errorText}>ℹ️  {result.error}</Text>
                    </View>
                )}

                {/* Key EEG Indicators */}
                <View style={styles.eegSection}>
                    <Text style={styles.eegTitle}>Key EEG Indicators</Text>
                    <View style={styles.eegRow}>
                        <Text style={styles.eegLabel}>Alpha Power (8-13Hz)</Text>
                        <IndicatorBar value={alphaPower} color="#22c55e" />
                    </View>
                    <View style={styles.eegRow}>
                        <Text style={styles.eegLabel}>Theta Ratio</Text>
                        <IndicatorBar value={thetaRatio} color={thetaRatio > 0.6 ? '#ef4444' : '#22c55e'} />
                    </View>
                </View>

                {/* Healthy Tips Section */}
                {result.tips && result.tips.length > 0 && (
                    <View style={styles.tipsSection}>
                        <View style={styles.tipsHeader}>
                            <Text style={styles.tipsIcon}>💡</Text>
                            <Text style={styles.tipsTitle}>Daily Brain Health Tips</Text>
                        </View>
                        {result.tips.map((tip, idx) => (
                            <View key={idx} style={styles.tipCard}>
                                <Text style={styles.tipText}>{tip}</Text>
                            </View>
                        ))}
                    </View>
                )}

                {/* New Analysis button */}
                <TouchableOpacity
                    style={styles.newAnalysisBtn}
                    onPress={() => navigation.popToTop()}
                    activeOpacity={0.85}
                >
                    <Text style={styles.newAnalysisBtnText}>→  New Analysis</Text>
                </TouchableOpacity>
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#0d1117' },
    statusRow: { paddingHorizontal: 20, paddingTop: 6, alignItems: 'center' },
    timeText: { color: '#d1d5db', fontSize: 13, fontWeight: '500' },

    loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    loadingText: { color: '#94a3b8', fontSize: 18, marginTop: 16, fontWeight: '600' },

    scroll: { padding: 20, paddingBottom: 50 },

    headerRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 18 },
    headerIcon: { fontSize: 20, marginRight: 8 },
    headerTitle: { fontSize: 22, fontWeight: '800', color: '#e2e8f0' },

    diagCard: {
        borderRadius: 12, borderWidth: 1.5,
        paddingVertical: 22, paddingHorizontal: 20,
        alignItems: 'center', marginBottom: 14,
    },
    diagLabel: { color: '#d97706', fontSize: 11, letterSpacing: 2, fontWeight: '700', marginBottom: 8 },
    diagValue: { fontSize: 26, fontWeight: '800', textAlign: 'center' },

    statsRow: { flexDirection: 'row', gap: 12, marginBottom: 12 },
    statCard: {
        flex: 1, backgroundColor: '#0f172a', borderRadius: 10,
        borderWidth: 1.5, paddingVertical: 16, alignItems: 'center',
    },
    statLabel: { color: '#94a3b8', fontSize: 10, letterSpacing: 1.5, fontWeight: '700', marginBottom: 6 },
    statValue: { fontSize: 22, fontWeight: '800' },

    infoRow: {
        backgroundColor: '#111827', borderRadius: 8,
        paddingVertical: 10, paddingHorizontal: 14, marginBottom: 12,
    },
    infoRowText: { color: '#6b7280', fontSize: 12, textAlign: 'center' },

    errorBanner: {
        backgroundColor: '#1c1917', borderRadius: 8,
        paddingVertical: 8, paddingHorizontal: 14, marginBottom: 12,
        borderLeftWidth: 3, borderLeftColor: '#f59e0b',
    },
    errorText: { color: '#fcd34d', fontSize: 12 },

    eegSection: {
        backgroundColor: '#111827', borderRadius: 10,
        padding: 16, borderWidth: 1, borderColor: '#1f2937', marginBottom: 18,
    },
    eegTitle: { color: '#e2e8f0', fontSize: 14, fontWeight: '700', marginBottom: 14 },
    eegRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12, gap: 12 },
    eegLabel: { color: '#94a3b8', fontSize: 12, width: 140 },

    newAnalysisBtn: {
        backgroundColor: '#1e3a5f', borderRadius: 10,
        paddingVertical: 18, alignItems: 'center',
        borderWidth: 1, borderColor: '#3b82f6',
    },
    newAnalysisBtnText: { color: '#93c5fd', fontSize: 16, fontWeight: '700' },

    tipsSection: {
        marginBottom: 26,
    },
    tipsHeader: {
        flexDirection: 'row', alignItems: 'center', marginBottom: 12,
    },
    tipsIcon: { fontSize: 18, marginRight: 8 },
    tipsTitle: { color: '#f8fafc', fontSize: 16, fontWeight: '700' },
    tipCard: {
        backgroundColor: '#1e293b', borderRadius: 10,
        padding: 14, marginBottom: 8,
        borderLeftWidth: 3, borderLeftColor: '#3b82f6',
    },
    tipText: { color: '#cbd5e1', fontSize: 13, lineHeight: 18 },
});
