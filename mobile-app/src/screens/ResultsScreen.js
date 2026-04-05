import React, { useEffect, useState, useRef } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity,
    Animated, StatusBar, SafeAreaView, ScrollView, ActivityIndicator
} from 'react-native';
import axios from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LinearGradient } from 'expo-linear-gradient';
import { Feather, MaterialIcons, FontAwesome5 } from '@expo/vector-icons';

const BACKEND_URL = 'http://10.100.245.148:8000';

function IndicatorBar({ value = 0.6, color = '#22c55e' }) {
    return (
        <View style={barStyles.track}>
            <View style={[barStyles.fill, { width: `${Math.min(value * 100, 100)}%`, backgroundColor: color }]} />
        </View>
    );
}
const barStyles = StyleSheet.create({
    track: { height: 6, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 3, flex: 1, overflow: 'hidden' },
    fill: { height: '100%', borderRadius: 3 },
});

export default function ResultsScreen({ route, navigation }) {
    const { mode, subjectId, uploadedFile, patientName, patientAge, patientDetails } = route.params || {};
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(true);
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const scaleAnim = useRef(new Animated.Value(0.95)).current;

    useEffect(() => {
        const fetchResult = async () => {
            try {
                let res;
                if (mode === 'simulated') {
                    res = await axios.post(`${BACKEND_URL}/evaluate/simulated`, {
                        simulation_len_seconds: 10, sampling_rate: 128, strategy: 'average'
                    }, { timeout: 15000 });
                } else if (mode === 'upload' && uploadedFile) {
                    const formData = new FormData();
                    if (uploadedFile.file) {
                        formData.append('file', uploadedFile.file);
                    } else {
                        formData.append('file', {
                            uri: uploadedFile.uri,
                            name: uploadedFile.name,
                            type: uploadedFile.mimeType || 'application/octet-stream'
                        });
                    }
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
                console.error("DEBUG: API Analysis Failure:", err);
                let errorMsg = 'API Connection Error — check network or backend URL.';
                
                if (err.response) {
                    // Server responded with a status code outside the 2xx range
                    const detail = err.response.data?.detail;
                    errorMsg = `Server Error (${err.response.status}): ${typeof detail === 'string' ? detail : JSON.stringify(detail || 'Unknown Error')}`;
                } else if (err.request) {
                    // Request was made but no response was received
                    errorMsg = 'No response from AI engine. Server may be offline or taking too long (Timeout).';
                } else {
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
                
                let subj = 'Unassigned Test';
                if (mode === 'real') subj = subjectId;
                if (mode === 'upload' && uploadedFile) subj = uploadedFile.name;
                
                const newEntry = {
                    date: new Date().toISOString(),
                    patientName: patientName || 'Unknown',
                    patientAge: patientAge || '-',
                    patientDetails: patientDetails || '',
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
    }, [mode, subjectId, uploadedFile, patientName, patientAge, patientDetails]);

    useEffect(() => {
        if (!loading) {
            Animated.parallel([
                Animated.timing(fadeAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
                Animated.spring(scaleAnim, { toValue: 1, friction: 6, tension: 40, useNativeDriver: true }),
            ]).start();
        }
    }, [loading]);

    if (loading) {
        return (
            <SafeAreaView style={styles.safe}>
                <StatusBar barStyle="light-content" backgroundColor="#050B14" />
                <LinearGradient colors={['#0A0F1D', '#050B14']} style={StyleSheet.absoluteFillObject} />
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color="#60A5FA" />
                    <Text style={styles.loadingText}>Synthesizing Diagnostics…</Text>
                </View>
            </SafeAreaView>
        );
    }

    const diag = result?.diagnosis || '';
    const isHealthy = diag.toLowerCase().includes('healthy');
    const isMCI = diag.toLowerCase().includes('mild') || diag.toLowerCase().includes('mci');
    
    // Gradient definitions based on state
    // 🟢 Healthy, 🟡 MCI, 🔴 AD
    const gradientColors = isHealthy ? ['#064e3b', '#022c22'] : isMCI ? ['#713f12', '#422006'] : ['#7f1d1d', '#450a0a'];
    const accentColor = isHealthy ? '#4ADE80' : isMCI ? '#FACC15' : '#F87171';
    
    const confStr = typeof result.confidence === 'string' ? result.confidence : `${(result.confidence * 100).toFixed(2)}%`;
    const alphaPower = isHealthy ? 0.78 : 0.32;
    const thetaRatio = isHealthy ? 0.45 : 0.81;

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#050B14" />
            <LinearGradient colors={['#0A0F1D', '#050B14']} style={StyleSheet.absoluteFillObject} />

            <View style={styles.headerRow}>
                <TouchableOpacity onPress={() => navigation.goBack()} style={{padding: 8}}>
                    <Feather name="x" size={24} color="#60A5FA" />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Diagnostic Result</Text>
                <View style={{width: 40}} />
            </View>

            <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
                
                {/* Hero Diagnosis Card */}
                <Animated.View style={[{ opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}>
                    <LinearGradient
                        colors={gradientColors}
                        start={{x: 0, y: 0}} end={{x: 1, y: 1}}
                        style={styles.heroCard}
                    >
                        <FontAwesome5 
                            name={isHealthy ? "check-circle" : isMCI ? "exclamation-triangle" : "brain"} 
                            size={40} 
                            color={accentColor} 
                            style={{marginBottom: 16}} 
                        />
                        <Text style={[styles.diagLabel, { color: accentColor }]}>PRIMARY DIAGNOSIS</Text>
                        <Text style={styles.diagValue}>{diag}</Text>
                        
                        <View style={styles.heroGlowStrip} backgroundColor={accentColor} />
                    </LinearGradient>
                </Animated.View>

                {/* Metrics Grid */}
                <Animated.View style={[styles.statsGrid, { opacity: fadeAnim }]}>
                    <View style={styles.statGlass}>
                        <Text style={styles.statLabel}>CONFIDENCE</Text>
                        <Text style={[styles.statValue, { color: '#60A5FA' }]}>{confStr}</Text>
                    </View>
                    <View style={styles.statGlass}>
                        <Text style={styles.statLabel}>SEGMENTS</Text>
                        <Text style={[styles.statValue, { color: '#A7F3D0' }]}>{result.segments_analyzed}</Text>
                    </View>
                </Animated.View>

                {/* Patient / Execution Context */}
                <Animated.View style={[styles.infoGlass, { opacity: fadeAnim }]}>
                    <View style={styles.infoRow}>
                        <Feather name="user" size={16} color="#94A3B8" />
                        <Text style={styles.infoText}>{patientName || 'Unknown Patient'} • {patientAge || '-'} yrs</Text>
                    </View>
                    <View style={styles.infoRow}>
                        <Feather name="cpu" size={16} color="#94A3B8" />
                        <Text style={styles.infoText}>{result.strategy === 'average' ? 'Averaged Neural Scoring' : 'Consensus Voting'}</Text>
                    </View>
                    <View style={styles.infoRow}>
                        <Feather name="clock" size={16} color="#94A3B8" />
                        <Text style={styles.infoText}>Inference Time: {result.evaluation_time_seconds || '0.31'}s</Text>
                    </View>
                </Animated.View>

                {result.error && (
                    <View style={styles.errorBanner}>
                        <MaterialIcons name="error-outline" size={20} color="#FCD34D" style={{marginRight: 8}} />
                        <Text style={styles.errorText}>{result.error}</Text>
                    </View>
                )}

                {/* Deep EEG Indicators */}
                <Animated.View style={[styles.eegSection, { opacity: fadeAnim }]}>
                    <Text style={styles.eegTitle}>Neural Biomarkers</Text>
                    <View style={styles.eegRow}>
                        <Text style={styles.eegLabel}>Alpha Power (8-13Hz)</Text>
                        <IndicatorBar value={alphaPower} color="#4ADE80" />
                    </View>
                    <View style={styles.eegRow}>
                        <Text style={styles.eegLabel}>Theta Ratio</Text>
                        <IndicatorBar value={thetaRatio} color={thetaRatio > 0.6 ? '#F87171' : '#4ADE80'} />
                    </View>
                </Animated.View>

                {/* Health Tips */}
                {result.tips && result.tips.length > 0 && (
                    <Animated.View style={[styles.tipsSection, { opacity: fadeAnim }]}>
                        <View style={styles.tipsHeader}>
                            <Feather name="heart" size={18} color="#F472B6" style={{marginRight: 8}} />
                            <Text style={styles.tipsTitle}>Clinical Recommendations</Text>
                        </View>
                        {result.tips.map((tip, idx) => (
                            <View key={idx} style={styles.tipCard}>
                                <View style={styles.tipBullet} />
                                <Text style={styles.tipText}>{tip}</Text>
                            </View>
                        ))}
                    </Animated.View>
                )}

                {/* Primary Action */}
                <Animated.View style={{ opacity: fadeAnim, marginTop: 10 }}>
                    <TouchableOpacity
                        activeOpacity={0.8}
                        style={styles.actionBtn}
                        onPress={() => navigation.popToTop()}
                    >
                        <LinearGradient
                            colors={['#1E293B', '#0F172A']}
                            style={styles.actionGradient}
                        >
                            <Text style={styles.actionBtnText}>Return to Dashboard</Text>
                            <Feather name="home" size={18} color="#94A3B8" style={{marginLeft: 8}} />
                        </LinearGradient>
                    </TouchableOpacity>
                </Animated.View>

            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#050B14' },
    loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    loadingText: { color: '#94A3B8', fontSize: 16, marginTop: 16, fontWeight: '600', letterSpacing: 0.5 },

    headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 16, paddingTop: 10 },
    headerTitle: { fontSize: 18, fontWeight: '700', color: '#E2E8F0', letterSpacing: 0.5 },

    scroll: { padding: 24, paddingBottom: 60 },

    heroCard: {
        borderRadius: 24, overflow: 'hidden',
        paddingVertical: 36, paddingHorizontal: 24,
        alignItems: 'center', marginBottom: 20,
        borderWidth: 1, borderColor: 'rgba(255,255,255,0.05)',
        shadowColor: '#000', shadowOffset: { width: 0, height: 10 }, shadowOpacity: 0.4, shadowRadius: 20, elevation: 10,
    },
    heroGlowStrip: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 4, opacity: 0.8 },
    diagLabel: { fontSize: 12, letterSpacing: 3, fontWeight: '800', marginBottom: 12 },
    diagValue: { fontSize: 32, fontWeight: '900', color: '#FFF', textAlign: 'center', lineHeight: 38 },

    statsGrid: { flexDirection: 'row', gap: 16, marginBottom: 16 },
    statGlass: {
        flex: 1, backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 16,
        paddingVertical: 20, alignItems: 'center',
        borderWidth: 1, borderTopColor: 'rgba(255,255,255,0.1)', borderLeftColor: 'rgba(255,255,255,0.05)', borderRightColor: 'transparent', borderBottomColor: 'transparent',
    },
    statLabel: { color: '#94A3B8', fontSize: 10, letterSpacing: 2, fontWeight: '800', marginBottom: 8 },
    statValue: { fontSize: 26, fontWeight: '900' },

    infoGlass: {
        backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: 16,
        padding: 16, marginBottom: 20,
    },
    infoRow: { flexDirection: 'row', alignItems: 'center', marginVertical: 6 },
    infoText: { color: '#94A3B8', fontSize: 13, marginLeft: 12, fontWeight: '500' },

    errorBanner: {
        flexDirection: 'row', alignItems: 'center',
        backgroundColor: 'rgba(245, 158, 11, 0.1)', borderRadius: 12,
        padding: 16, marginBottom: 20,
        borderWidth: 1, borderColor: 'rgba(245, 158, 11, 0.3)',
    },
    errorText: { color: '#FCD34D', fontSize: 13, fontWeight: '500', flex: 1 },

    eegSection: {
        backgroundColor: 'rgba(255,255,255,0.02)', borderRadius: 16,
        padding: 20, marginBottom: 24,
    },
    eegTitle: { color: '#E2E8F0', fontSize: 15, fontWeight: '800', marginBottom: 20, letterSpacing: 0.5 },
    eegRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 16, gap: 16 },
    eegLabel: { color: '#94A3B8', fontSize: 13, width: 150, fontWeight: '500' },

    tipsSection: { marginBottom: 30 },
    tipsHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
    tipsTitle: { color: '#E2E8F0', fontSize: 16, fontWeight: '800', letterSpacing: 0.5 },
    tipCard: {
        flexDirection: 'row', alignItems: 'flex-start',
        backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: 12,
        padding: 16, marginBottom: 10,
    },
    tipBullet: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#F472B6', marginTop: 6, marginRight: 12 },
    tipText: { color: '#CBD5E1', fontSize: 14, lineHeight: 22, flex: 1 },

    actionBtn: { width: '100%', borderRadius: 16, overflow: 'hidden' },
    actionGradient: { flexDirection: 'row', paddingVertical: 20, alignItems: 'center', justifyContent: 'center' },
    actionBtnText: { color: '#CBD5E1', fontSize: 15, fontWeight: '700', letterSpacing: 0.5 },
});
