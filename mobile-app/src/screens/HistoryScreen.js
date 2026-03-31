import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, FlatList, TouchableOpacity, SafeAreaView, StatusBar, ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

export default function HistoryScreen({ navigation }) {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const loadHistory = async () => {
            try {
                const jsonValue = await AsyncStorage.getItem('@mtdnet_history');
                const data = jsonValue != null ? JSON.parse(jsonValue) : [];
                // Sort by newest first
                data.sort((a, b) => new Date(b.date) - new Date(a.date));
                setHistory(data);
            } catch (e) {
                console.error("Failed to load history", e);
            } finally {
                setLoading(false);
            }
        };
        loadHistory();
    }, []);

    const clearHistory = async () => {
        try {
            await AsyncStorage.removeItem('@mtdnet_history');
            setHistory([]);
        } catch (e) {
            console.error("Failed to clear history", e);
        }
    };

    const renderItem = ({ item }) => {
        const isHealthy = item.diagnosis.toLowerCase().includes('healthy');
        const color = isHealthy ? '#22c55e' : '#ef4444';
        
        return (
            <View style={styles.card}>
                <View style={styles.cardHeader}>
                    <Text style={styles.subjectText}>{item.subject || 'Analysis'}</Text>
                    <Text style={styles.dateText}>{new Date(item.date).toLocaleDateString()} {new Date(item.date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</Text>
                </View>
                <View style={styles.cardBody}>
                    <View style={styles.diagBox}>
                        <Text style={styles.label}>DIAGNOSIS</Text>
                        <Text style={[styles.diagnosisText, { color }]}>{item.diagnosis}</Text>
                    </View>
                    <View style={styles.statBox}>
                        <Text style={styles.label}>PERCENTAGE</Text>
                        <Text style={styles.confidenceText}>{item.confidence}</Text>
                    </View>
                </View>
            </View>
        );
    };

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#0d1117" />
            <View style={styles.headerRow}>
                <TouchableOpacity onPress={() => navigation.goBack()}>
                    <Text style={styles.backBtn}>◀  Back</Text>
                </TouchableOpacity>
                <Text style={styles.headerTitle}>Usage History</Text>
                <TouchableOpacity onPress={clearHistory}>
                    <Text style={styles.clearBtn}>Clear</Text>
                </TouchableOpacity>
            </View>

            {loading ? (
                <View style={styles.center}>
                    <ActivityIndicator size="large" color="#3b5bdb" />
                </View>
            ) : history.length === 0 ? (
                <View style={styles.center}>
                    <Text style={styles.emptyText}>No previous analyses found.</Text>
                </View>
            ) : (
                <FlatList
                    data={history}
                    keyExtractor={(item, index) => index.toString()}
                    renderItem={renderItem}
                    contentContainerStyle={styles.listContent}
                    showsVerticalScrollIndicator={false}
                />
            )}
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    safe: { flex: 1, backgroundColor: '#0d1117' },
    headerRow: {
        flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
        paddingHorizontal: 20, paddingVertical: 16,
        borderBottomWidth: 1, borderBottomColor: '#1f2937'
    },
    backBtn: { color: '#93c5fd', fontSize: 16, fontWeight: '600' },
    clearBtn: { color: '#ef4444', fontSize: 14, fontWeight: '600' },
    headerTitle: { color: '#f8fafc', fontSize: 18, fontWeight: '700' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    emptyText: { color: '#94a3b8', fontSize: 16 },
    listContent: { padding: 20 },
    card: {
        backgroundColor: '#111827', borderRadius: 12, borderWidth: 1, borderColor: '#1f2937',
        padding: 16, marginBottom: 16
    },
    cardHeader: {
        flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
        marginBottom: 12, paddingBottom: 10, borderBottomWidth: 1, borderBottomColor: '#1f2937'
    },
    subjectText: { color: '#e2e8f0', fontSize: 15, fontWeight: '700' },
    dateText: { color: '#64748b', fontSize: 12 },
    cardBody: { flexDirection: 'row', justifyContent: 'space-between' },
    diagBox: { flex: 2 },
    statBox: { flex: 1, alignItems: 'flex-end' },
    label: { color: '#64748b', fontSize: 10, letterSpacing: 1, marginBottom: 4, fontWeight: '700' },
    diagnosisText: { fontSize: 16, fontWeight: '800' },
    confidenceText: { color: '#60a5fa', fontSize: 18, fontWeight: '800' }
});
