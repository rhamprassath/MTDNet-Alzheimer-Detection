import React, { useEffect, useRef, useState } from 'react';
import {
    View, Text, StyleSheet, TouchableOpacity,
    Animated, StatusBar, SafeAreaView, Dimensions
} from 'react-native';
import axios from 'axios';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons, Feather } from '@expo/vector-icons';

const BACKEND_URL = 'http://10.100.245.148:8000'; // Local API URL
const { width } = Dimensions.get('window');

export default function HomeScreen({ navigation }) {
    const [apiStatus, setApiStatus] = useState('checking'); // 'checking' | 'connected' | 'error'
    const fadeAnim = useRef(new Animated.Value(0)).current;
    const pulseAnim = useRef(new Animated.Value(1)).current;

    useEffect(() => {
        Animated.timing(fadeAnim, {
            toValue: 1, duration: 1000, useNativeDriver: true,
        }).start();

        axios.get(`${BACKEND_URL}/health`, { timeout: 4000 })
            .then(() => setApiStatus('connected'))
            .catch(() => setApiStatus('error'));

        Animated.loop(
            Animated.sequence([
                Animated.timing(pulseAnim, { toValue: 1.25, duration: 1200, useNativeDriver: true }),
                Animated.timing(pulseAnim, { toValue: 1, duration: 1200, useNativeDriver: true }),
            ])
        ).start();
    }, []);

    const isConnected = apiStatus === 'connected';

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#050B14" />
            
            {/* Background Base gradient for depth */}
            <LinearGradient
                colors={['#050B14', '#0A0F1D']}
                style={StyleSheet.absoluteFillObject}
            />

            <Animated.View style={[styles.container, { opacity: fadeAnim }]}>
                
                {/* Status Indicator */}
                <View style={[styles.statusGlass, { borderColor: isConnected ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)' }]}>
                    <Animated.View style={[
                        styles.statusDot, 
                        { backgroundColor: isConnected ? '#4ade80' : '#ef4444' },
                        { transform: [{ scale: pulseAnim }], opacity: isConnected ? 1 : 0.7 }
                    ]} />
                    <Text style={[styles.statusText, { color: isConnected ? '#a7f3d0' : '#fca5a5' }]}>
                        {isConnected ? 'Core AI Online' : 'Core AI Offline'}
                    </Text>
                </View>

                {/* Hero Logo Section */}
                <View style={styles.heroGlow}>
                    <LinearGradient
                        colors={['rgba(59,130,246,0.15)', 'rgba(139,92,246,0.15)']}
                        style={styles.logoOuter}
                    >
                        <View style={styles.logoInner}>
                            <MaterialCommunityIcons name="brain" size={48} color="#60A5FA" />
                        </View>
                    </LinearGradient>
                </View>

                {/* Typography */}
                <Text style={styles.brandTitle}>MTDNet</Text>
                <Text style={styles.brandSubtitle}>Clinical Alzheimer's Diagnostics</Text>

                <View style={styles.cardsContainer}>
                    {/* Primary Action Button (The Gradients) */}
                    <TouchableOpacity
                        activeOpacity={0.85}
                        style={styles.actionTouchable}
                        onPress={() => navigation.navigate('Analysis', { mode: 'upload' })}
                    >
                        <LinearGradient
                            colors={['#2563EB', '#4F46E5']}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 1 }}
                            style={styles.primaryCard}
                        >
                            <View style={styles.cardHeaderRow}>
                                <Feather name="activity" size={24} color="#FFF" />
                                <Feather name="arrow-right" size={20} color="rgba(255,255,255,0.7)" />
                            </View>
                            <View style={styles.cardTextContent}>
                                <Text style={styles.primaryTitle}>Run New Scan</Text>
                                <Text style={styles.primaryDesc}>Upload EDF/CSV/SET files for deep neural evaluation.</Text>
                            </View>
                        </LinearGradient>
                    </TouchableOpacity>

                    {/* Secondary Actions Row */}
                    <View style={styles.secondaryRow}>
                        <TouchableOpacity
                            activeOpacity={0.7}
                            style={[styles.glassCard, { flex: 1, marginRight: 8 }]}
                            onPress={() => navigation.navigate('History')}
                        >
                            <Feather name="clock" size={24} color="#93C5FD" marginBottom={12} />
                            <Text style={styles.glassTitle}>History</Text>
                            <Text style={styles.glassDesc}>View patient records</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            activeOpacity={0.7}
                            style={[styles.glassCard, { flex: 1, marginLeft: 8 }]}
                            onPress={() => navigation.navigate('Analysis', { mode: 'real' })}
                        >
                            <Feather name="database" size={24} color="#A7F3D0" marginBottom={12} />
                            <Text style={styles.glassTitle}>Database</Text>
                            <Text style={styles.glassDesc}>Simulate subjects</Text>
                        </TouchableOpacity>
                    </View>
                </View>

            </Animated.View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#050B14' },
    
    container: { 
        flex: 1, 
        alignItems: 'center', 
        paddingHorizontal: 24,
        paddingTop: 60,
    },

    statusGlass: {
        flexDirection: 'row', alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.03)',
        borderWidth: 1,
        borderRadius: 30, paddingHorizontal: 16, paddingVertical: 8,
        marginBottom: 40,
    },
    statusDot: { width: 8, height: 8, borderRadius: 4, marginRight: 8, shadowColor: '#4ade80', shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.8, shadowRadius: 6, elevation: 4 },
    statusText: { fontSize: 13, fontWeight: '700', letterSpacing: 0.5 },

    heroGlow: {
        shadowColor: '#3B82F6', shadowOffset: { width: 0, height: 0 }, shadowOpacity: 0.4, shadowRadius: 40, elevation: 10,
        marginBottom: 24,
    },
    logoOuter: {
        width: 130, height: 130, borderRadius: 65,
        justifyContent: 'center', alignItems: 'center',
        borderWidth: 1, borderColor: 'rgba(96,165,250,0.3)',
    },
    logoInner: {
        width: 80, height: 80, borderRadius: 40,
        backgroundColor: '#0F172A',
        justifyContent: 'center', alignItems: 'center',
        borderWidth: 1, borderColor: '#1E293B',
    },

    brandTitle: { fontSize: 38, fontWeight: '900', color: '#F8FAFC', letterSpacing: 1.5 },
    brandSubtitle: { fontSize: 14, color: '#94A3B8', fontWeight: '500', marginTop: 4, letterSpacing: 1, textTransform: 'uppercase' },

    cardsContainer: { width: '100%', marginTop: 50 },

    actionTouchable: { width: '100%', marginBottom: 16 },
    primaryCard: {
        width: '100%', borderRadius: 24, padding: 24,
        shadowColor: '#4F46E5', shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.4, shadowRadius: 20, elevation: 8,
    },
    cardHeaderRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
    cardTextContent: {},
    primaryTitle: { fontSize: 24, fontWeight: '800', color: '#FFF', marginBottom: 6 },
    primaryDesc: { fontSize: 13, color: 'rgba(255,255,255,0.8)', lineHeight: 20 },

    secondaryRow: { flexDirection: 'row', width: '100%', justifyContent: 'space-between' },
    glassCard: {
        backgroundColor: 'rgba(15,23,42,0.6)', 
        borderWidth: 1, borderColor: 'rgba(51,65,85,0.8)',
        borderRadius: 20, padding: 20,
    },
    glassTitle: { fontSize: 16, fontWeight: '700', color: '#E2E8F0', marginBottom: 4 },
    glassDesc: { fontSize: 12, color: '#94A3B8' },
});
