import React, { useEffect, useRef, useState } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity,
    Animated, StatusBar, SafeAreaView
} from 'react-native';
import axios from 'axios';

const BACKEND_URL = 'http://192.168.29.19:8000';

export default function HomeScreen({ navigation }) {
    const [apiStatus, setApiStatus] = useState('checking'); // 'checking' | 'connected' | 'error'
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const pulseAnim = useRef(new Animated.Value(1)).current;

    useEffect(() => {
        Animated.timing(fadeAnim, {
            toValue: 1, duration: 900, useNativeDriver: true,
        }).start();

        // Ping the health endpoint
        axios.get(`${BACKEND_URL}/health`, { timeout: 4000 })
            .then(() => setApiStatus('connected'))
            .catch(() => setApiStatus('error'));

        // Pulse animation for the API dot
        Animated.loop(
            Animated.sequence([
                Animated.timing(pulseAnim, { toValue: 1.4, duration: 900, useNativeDriver: true }),
                Animated.timing(pulseAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
            ])
        ).start();
    }, []);

    const apiDotColor = apiStatus === 'connected' ? '#4ade80' : apiStatus === 'error' ? '#ef4444' : '#facc15';
    const apiLabel = apiStatus === 'connected' ? 'API Connected' : apiStatus === 'error' ? 'API Error' : 'Connecting…';

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#0d1117" />
            {/* Status bar row */}
            <View style={styles.statusRow}>
                <Text style={styles.timeText}>9:41</Text>
            </View>

            <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
                {/* Logo circle */}
                <View style={styles.logoOuter}>
                    <View style={styles.logoInner}>
                        <Text style={styles.logoIcon}>⬛</Text>
                    </View>
                </View>

                {/* Brand name */}
                <Text style={styles.brandTitle}>MTDNet</Text>
                <Text style={styles.brandSubtitle}>Early Alzheimer's{'\n'}Risk Detection</Text>

                {/* API Status badge */}
                <View style={styles.apiBadge}>
                    <Animated.View style={[styles.apiDot, { backgroundColor: apiDotColor, transform: [{ scale: pulseAnim }] }]} />
                    <Text style={styles.apiLabel}>{apiLabel}</Text>
                    <Text style={styles.apiIcon}>  ⟩</Text>
                </View>

                {/* NEW Primary button – Upload TEST Data */}
                <TouchableOpacity
                    style={styles.primaryBtn}
                    activeOpacity={0.8}
                    onPress={() => navigation.navigate('Analysis', { mode: 'upload' })}
                >
                    <Text style={styles.primaryBtnText}>🧪  Test EEG Signal (EDF/CSV)</Text>
                </TouchableOpacity>

                {/* History Button (Moved up slightly) */}
                <TouchableOpacity
                    style={styles.historyBtn}
                    activeOpacity={0.8}
                    onPress={() => navigation.navigate('History')}
                >
                    <Text style={styles.historyBtnText}>🕒  View Usage History</Text>
                </TouchableOpacity>

                <View style={styles.divider} />

                {/* Secondary buttons – Simulated & Real */}
                <TouchableOpacity
                    style={styles.secondaryBtn}
                    activeOpacity={0.8}
                    onPress={() => navigation.navigate('Analysis', { mode: 'simulated' })}
                >
                    <Text style={styles.secondaryBtnText}>⚡  Simulated Stream Analysis</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.secondaryBtn}
                    activeOpacity={0.8}
                    onPress={() => navigation.navigate('Analysis', { mode: 'real' })}
                >
                    <Text style={styles.secondaryBtnText}>👤  Real Subject Database</Text>
                </TouchableOpacity>

                {/* Model info footer */}
                <View style={styles.infoBox}>
                    <Text style={styles.infoLine}>Model: MTDNet Ensemble</Text>
                    <Text style={styles.infoLine}>Accuracy: 89.23%</Text>
                    <Text style={styles.infoLine}>Format Support: EDF, CSV, SET</Text>
                </View>
            </Animated.View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#0d1117' },
    statusRow: { paddingHorizontal: 20, paddingTop: 6, alignItems: 'center' },
    timeText: { color: '#d1d5db', fontSize: 13, fontWeight: '500' },

    container: { flex: 1, alignItems: 'center', paddingHorizontal: 24, paddingTop: 10 },

    logoOuter: {
        width: 140, height: 140, borderRadius: 70,
        borderWidth: 2, borderColor: '#3b5bdb',
        justifyContent: 'center', alignItems: 'center',
        backgroundColor: 'rgba(59,91,219,0.08)',
        marginBottom: 18,
    },
    logoInner: {
        width: 70, height: 70, borderRadius: 35,
        backgroundColor: '#1e2a4a',
        justifyContent: 'center', alignItems: 'center',
    },
    logoIcon: { fontSize: 28 },

    brandTitle: { fontSize: 32, fontWeight: '800', color: '#e2e8f0', letterSpacing: 1 },
    brandSubtitle: { fontSize: 14, color: '#94a3b8', textAlign: 'center', marginTop: 4, lineHeight: 20, marginBottom: 16 },

    apiBadge: {
        flexDirection: 'row', alignItems: 'center',
        borderWidth: 1, borderColor: '#22c55e',
        borderRadius: 20, paddingHorizontal: 14, paddingVertical: 6,
        marginBottom: 28,
    },
    apiDot: { width: 8, height: 8, borderRadius: 4, marginRight: 7 },
    apiLabel: { color: '#86efac', fontSize: 13, fontWeight: '600' },
    apiIcon: { color: '#86efac', fontSize: 13 },

    primaryBtn: {
        width: '100%', backgroundColor: '#3b5bdb',
        paddingVertical: 18, borderRadius: 10, alignItems: 'center', marginBottom: 12,
    },
    primaryBtnText: { color: '#fff', fontSize: 16, fontWeight: '700' },

    secondaryBtn: {
        width: '100%', backgroundColor: '#1a4731',
        paddingVertical: 18, borderRadius: 10, alignItems: 'center', marginBottom: 12,
        borderWidth: 1, borderColor: '#22c55e',
    },
    secondaryBtnText: { color: '#86efac', fontSize: 16, fontWeight: '700' },

    historyBtn: {
        width: '100%', backgroundColor: 'transparent',
        paddingVertical: 14, borderRadius: 10, alignItems: 'center', marginBottom: 22,
        borderWidth: 1, borderColor: '#3b82f6',
    },
    historyBtnText: { color: '#93c5fd', fontSize: 16, fontWeight: '700' },

    divider: {
        width: '100%', height: 1, backgroundColor: 'rgba(255,255,255,0.1)',
        marginVertical: 15,
    },

    infoBox: {
        width: '100%', backgroundColor: '#111827',
        borderRadius: 10, padding: 16,
        borderWidth: 1, borderColor: '#1f2937',
    },
    infoLine: { color: '#6b7280', fontSize: 13, marginVertical: 3, textAlign: 'center' },
});
