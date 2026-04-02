import React, { useEffect, useState } from 'react';
import { 
    View, Text, StyleSheet, FlatList, TouchableOpacity, 
    SafeAreaView, StatusBar, ActivityIndicator, Dimensions, TextInput 
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LinearGradient } from 'expo-linear-gradient';
import { Feather, MaterialCommunityIcons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

export default function HistoryScreen({ navigation }) {
    const [history, setHistory] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');

    useEffect(() => {
        const loadHistory = async () => {
            try {
                const jsonValue = await AsyncStorage.getItem('@mtdnet_history');
                let data = jsonValue != null ? JSON.parse(jsonValue) : [];
                // Sort by newest first
                data.sort((a, b) => new Date(b.date) - new Date(a.date));
                setHistory(data);
            } catch (e) {
                console.error("Failed to load history", e);
            } finally {
                setLoading(false);
            }
        };
        const unsubscribe = navigation.addListener('focus', loadHistory);
        loadHistory();
        return unsubscribe;
    }, [navigation]);

    const filteredHistory = history.filter(item => 
        (item.patientName || '').toLowerCase().includes(searchQuery.toLowerCase())
    );

    const clearHistory = async () => {
        try {
            await AsyncStorage.removeItem('@mtdnet_history');
            setHistory([]);
        } catch (e) {
            console.error("Failed to clear history", e);
        }
    };

    const renderItem = ({ item, index }) => {
        const diag = item.diagnosis || '';
        const isHealthy = diag.toLowerCase().includes('healthy');
        const isMCI = diag.toLowerCase().includes('mild') || diag.toLowerCase().includes('mci');
        
        // 🟢 Healthy, 🟡 MCI, 🔴 AD
        const statusColor = isHealthy ? '#4ADE80' : isMCI ? '#FACC15' : '#F87171';
        const dateObj = new Date(item.date);
        
        return (
            <View style={styles.historyItemNode}>
                {/* Timeline vertical bar */}
                <View style={styles.timelineContainer}>
                    <View style={[styles.timelineDot, { backgroundColor: statusColor }]} />
                    {index < history.length - 1 && <View style={styles.timelineLine} />}
                </View>

                {/* Glass Card */}
                <View style={styles.glassCard}>
                    <View style={styles.cardHeader}>
                        <View style={styles.patientInfo}>
                            <Text style={styles.patientName}>{item.patientName || 'Anonymous'}</Text>
                            <Text style={styles.patientMeta}>
                                {item.patientAge ? `${item.patientAge}y` : 'N/A'} • {item.subject || 'Simulation'}
                            </Text>
                        </View>
                        <View style={styles.dateInfo}>
                            <Text style={styles.dateDay}>{dateObj.getDate()}</Text>
                            <Text style={styles.dateMonth}>{dateObj.toLocaleString('default', { month: 'short' })}</Text>
                        </View>
                    </View>

                    {item.patientDetails ? (
                        <View style={styles.detailsBox}>
                            <Feather name="file-text" size={12} color="#64748B" />
                            <Text style={styles.detailsText} numberOfLines={1}>{item.patientDetails}</Text>
                        </View>
                    ) : null}

                    <View style={styles.diagContainer}>
                        <View style={styles.diagTextWrapper}>
                            <Text style={styles.diagLabel}>DIAGNOSIS</Text>
                            <Text style={[styles.diagValue, { color: statusColor }]}>{diag}</Text>
                        </View>
                        <View style={styles.confWrapper}>
                            <Text style={styles.diagLabel}>CONFIDENCE</Text>
                            <Text style={styles.confValue}>{item.confidence}</Text>
                        </View>
                    </View>
                </View>
            </View>
        );
    };

    return (
        <SafeAreaView style={styles.safe}>
            <StatusBar barStyle="light-content" backgroundColor="#050B14" />
            <LinearGradient colors={['#0A0F1D', '#050B14']} style={StyleSheet.absoluteFillObject} />

            <View style={styles.headerContainer}>
                <View style={styles.headerRow}>
                    <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backBtn}>
                        <Feather name="chevron-left" size={24} color="#60A5FA" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>Neural History</Text>
                    <TouchableOpacity onPress={clearHistory} style={styles.clearBtn}>
                        <Feather name="trash-2" size={18} color="#F87171" />
                    </TouchableOpacity>
                </View>

                {/* Search Bar */}
                <View style={styles.searchContainer}>
                    <Feather name="search" size={16} color="#475569" style={styles.searchIcon} />
                    <TextInput
                        style={styles.searchInput}
                        placeholder="Search patient name (e.g. Ravi)..."
                        placeholderTextColor="#475569"
                        value={searchQuery}
                        onChangeText={setSearchQuery}
                        clearButtonMode="while-editing"
                    />
                </View>
            </View>

            {loading ? (
                <View style={styles.center}>
                    <ActivityIndicator size="large" color="#3B82F6" />
                </View>
            ) : filteredHistory.length === 0 ? (
                <View style={styles.center}>
                    <MaterialCommunityIcons name="database-off-outline" size={60} color="#1E293B" />
                    <Text style={styles.emptyText}>
                        {searchQuery ? `No records found for "${searchQuery}"` : 'No telemetry recorded.'}
                    </Text>
                </View>
            ) : (
                <FlatList
                    data={filteredHistory}
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
    safe: { flex: 1, backgroundColor: '#050B14' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    
    headerContainer: {
        paddingBottom: 8,
    },
    headerRow: {
        flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
        paddingHorizontal: 16, paddingVertical: 16,
    },
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.03)',
        marginHorizontal: 16,
        paddingHorizontal: 12,
        borderRadius: 12,
        borderWidth: 1,
        borderColor: '#1E293B',
        marginBottom: 8,
    },
    searchIcon: {
        marginRight: 8,
    },
    searchInput: {
        flex: 1,
        color: '#F8FAFC',
        height: 44,
        fontSize: 14,
    },
    backBtn: { padding: 4 },
    clearBtn: { padding: 4 },
    headerTitle: { color: '#F8FAFC', fontSize: 20, fontWeight: '800', letterSpacing: 0.5 },
    
    emptyText: { color: '#475569', fontSize: 16, marginTop: 16, fontWeight: '600' },
    
    listContent: { paddingHorizontal: 16, paddingBottom: 40 },
    
    historyItemNode: {
        flexDirection: 'row',
        marginBottom: 8,
    },
    
    timelineContainer: {
        width: 30,
        alignItems: 'center',
    },
    timelineDot: {
        width: 12, height: 12, borderRadius: 6,
        marginTop: 22,
        zIndex: 2,
        shadowOpacity: 0.5, shadowRadius: 4, elevation: 5,
    },
    timelineLine: {
        position: 'absolute',
        top: 30, bottom: -10,
        width: 1.5,
        backgroundColor: '#1E293B',
    },
    
    glassCard: {
        flex: 1,
        backgroundColor: 'rgba(255,255,255,0.025)',
        borderRadius: 20,
        borderWidth: 1, borderColor: 'rgba(255,255,255,0.05)',
        padding: 16,
        marginBottom: 16,
    },
    
    cardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
    },
    patientName: { color: '#F8FAFC', fontSize: 18, fontWeight: '800' },
    patientMeta: { color: '#64748B', fontSize: 12, fontWeight: '600', textTransform: 'uppercase', marginTop: 2 },
    
    dateInfo: { alignItems: 'center', backgroundColor: '#111827', paddingHorizontal: 10, paddingVertical: 6, borderRadius: 10, borderWidth: 1, borderColor: '#1E293B' },
    dateDay: { color: '#F8FAFC', fontSize: 14, fontWeight: '800' },
    dateMonth: { color: '#60A5FA', fontSize: 10, fontWeight: '800', textTransform: 'uppercase' },

    detailsBox: {
        flexDirection: 'row', alignItems: 'center',
        backgroundColor: 'rgba(0,0,0,0.2)',
        padding: 8, borderRadius: 8,
        marginBottom: 16,
    },
    detailsText: { color: '#94A3B8', fontSize: 12, marginLeft: 6 },

    diagContainer: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        backgroundColor: 'rgba(255,255,255,0.015)',
        padding: 12, borderRadius: 12,
        borderWidth: 1, borderColor: 'rgba(255,255,255,0.02)',
    },
    diagTextWrapper: { flex: 1.5 },
    confWrapper: { flex: 1, alignItems: 'flex-end' },
    
    diagLabel: { color: '#475569', fontSize: 9, fontWeight: '800', letterSpacing: 1, marginBottom: 4 },
    diagValue: { fontSize: 16, fontWeight: '900' },
    confValue: { color: '#60A5FA', fontSize: 16, fontWeight: '900' },
});
