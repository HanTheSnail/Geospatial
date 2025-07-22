import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import re

st.set_page_config(page_title="Secret Shopper Distance Analyzer", page_icon="🛍️", layout="wide")

@st.cache_resource
def get_geocoder():
    return Nominatim(user_agent="secret_shopper_analyzer")

def clean_address(address):
    """Clean and standardize address format"""
    if pd.isna(address) or not isinstance(address, str):
        return ""
    address = re.sub(r'\s+', ' ', address.strip())
    address = re.sub(r'\b(store|shop|location|unit|suite|ste|apt)\s*#?\s*\w*\b', '', address, flags=re.IGNORECASE)
    return address.strip()

def geocode_addresses(addresses, label="addresses"):
    """Geocode a list of addresses with progress tracking"""
    geolocator = get_geocoder()
    st.info(f"🗺️ Geocoding {len(addresses)} {label}...")
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, addr in enumerate(addresses):
        clean_addr = clean_address(addr)
        
        try:
            location = geolocator.geocode(clean_addr, timeout=10)
            coords = (location.latitude, location.longitude) if location else None
            time.sleep(0.3)  # Be gentle with free service
        except:
            coords = None
        
        results.append(coords)
        progress_bar.progress((i + 1) / len(addresses))
        status_text.text(f"Processed {i + 1}/{len(addresses)} {label}")
    
    success_count = sum(1 for r in results if r)
    st.success(f"✅ {success_count}/{len(addresses)} {label} geocoded successfully!")
    return results

def combine_address_columns(df, columns):
    """Combine multiple columns into addresses"""
    combined = []
    for _, row in df.iterrows():
        parts = []
        for col in columns:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                parts.append(str(row[col]).strip())
        combined.append(', '.join(parts))
    return combined

def calculate_distances(stores_df, users_df, max_km):
    """Calculate distances between all user-store pairs"""
    st.info("📏 Calculating distances...")
    
    # Filter for valid coordinates
    valid_stores = stores_df.dropna(subset=['latitude', 'longitude']).copy()
    valid_users = users_df.dropna(subset=['latitude', 'longitude']).copy()
    
    if valid_stores.empty or valid_users.empty:
        st.error("No valid geocoded data found!")
        return pd.DataFrame()
    
    results = []
    total_combinations = len(valid_users) * len(valid_stores)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    counter = 0
    for u_idx, user in valid_users.iterrows():
        user_coords = (user['latitude'], user['longitude'])
        
        for s_idx, store in valid_stores.iterrows():
            store_coords = (store['latitude'], store['longitude'])
            distance_km = geodesic(user_coords, store_coords).kilometers
            
            if distance_km <= max_km:
                # Create result record
                result = {
                    'user_idx': u_idx,
                    'store_idx': s_idx,
                    'distance_km': round(distance_km, 2),
                    'distance_miles': round(distance_km * 0.621371, 2)
                }
                
                # Add all user data
                for col in valid_users.columns:
                    result[f'user_{col}'] = user[col]
                
                # Add all store data
                for col in valid_stores.columns:
                    result[f'store_{col}'] = store[col]
                
                results.append(result)
            
            counter += 1
            if counter % 50 == 0:  # Update less frequently
                progress_bar.progress(counter / total_combinations)
                status_text.text(f"Calculated {counter}/{total_combinations} distances")
    
    st.success(f"✅ Found {len(results)} eligible user-store pairs within {max_km}km!")
    return pd.DataFrame(results)

def main():
    st.title("🛍️ Secret Shopper Distance Analyzer")
    st.markdown("Upload your data to find eligible participants for secret shopping campaigns.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        max_distance = st.slider("Maximum Distance (km)", min_value=1, max_value=200, value=50)
        
        st.markdown("---")
        st.markdown("""
        **How to use:**
        1. Upload store data
        2. Upload user data  
        3. Select address columns
        4. Process both datasets
        5. Calculate distances
        6. Export results
        """)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📊 Data Upload", "📋 Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        # STORE DATA
        with col1:
            st.subheader("🏪 Store Data")
            stores_file = st.file_uploader(
                "Upload store list (CSV/Excel)", 
                type=['csv', 'xlsx'], 
                key="stores_file"
            )
            
            if stores_file:
                # Load data
                if stores_file.name.endswith('.csv'):
                    stores_df = pd.read_csv(stores_file)
                else:
                    stores_df = pd.read_excel(stores_file)
                
                st.write("**Preview:**")
                st.dataframe(stores_df.head(), use_container_width=True)
                
                # Address configuration
                st.write("**Address Setup:**")
                address_type = st.radio(
                    "How is your store address formatted?",
                    ["Single column", "Multiple columns"],
                    key="store_address_type"
                )
                
                if address_type == "Single column":
                    address_col = st.selectbox("Select address column:", stores_df.columns, key="store_addr_col")
                    addresses = stores_df[address_col].tolist()
                else:
                    address_cols = st.multiselect("Select columns to combine:", stores_df.columns, key="store_addr_cols")
                    if address_cols:
                        # Show preview
                        preview = stores_df[address_cols].head(2)
                        for i, row in preview.iterrows():
                            combined = ', '.join([str(row[c]) for c in address_cols if pd.notna(row[c])])
                            st.write(f"Preview {i+1}: `{combined}`")
                        addresses = combine_address_columns(stores_df, address_cols)
                    else:
                        addresses = []
                
                # Process button
                if st.button("🔄 Process Store Data", type="primary"):
                    if addresses:
                        coords = geocode_addresses(addresses, "stores")
                        stores_df[['latitude', 'longitude']] = coords
                        st.session_state['stores_df'] = stores_df
                        st.session_state['stores_ready'] = True
                        st.rerun()
                    else:
                        st.error("Please select address column(s) first!")
        
        # USER DATA  
        with col2:
            st.subheader("👥 User Data")
            users_file = st.file_uploader(
                "Upload user list (CSV/Excel)", 
                type=['csv', 'xlsx'], 
                key="users_file"
            )
            
            if users_file:
                # Load data
                if users_file.name.endswith('.csv'):
                    users_df = pd.read_csv(users_file)
                else:
                    users_df = pd.read_excel(users_file)
                
                st.write("**Preview:**")
                st.dataframe(users_df.head(), use_container_width=True)
                
                # Address configuration
                st.write("**Address Setup:**")
                address_type = st.radio(
                    "How is your user address formatted?",
                    ["Single column", "Multiple columns"],
                    key="user_address_type"
                )
                
                if address_type == "Single column":
                    address_col = st.selectbox("Select address column:", users_df.columns, key="user_addr_col")
                    addresses = users_df[address_col].tolist()
                else:
                    address_cols = st.multiselect("Select columns to combine:", users_df.columns, key="user_addr_cols")
                    if address_cols:
                        # Show preview
                        preview = users_df[address_cols].head(2)
                        for i, row in preview.iterrows():
                            combined = ', '.join([str(row[c]) for c in address_cols if pd.notna(row[c])])
                            st.write(f"Preview {i+1}: `{combined}`")
                        addresses = combine_address_columns(users_df, address_cols)
                    else:
                        addresses = []
                
                # Process button
                if st.button("🔄 Process User Data", type="primary"):
                    if addresses:
                        coords = geocode_addresses(addresses, "users")
                        users_df[['latitude', 'longitude']] = coords
                        st.session_state['users_df'] = users_df
                        st.session_state['users_ready'] = True
                        st.rerun()
                    else:
                        st.error("Please select address column(s) first!")
    
    with tab2:
        st.header("Distance Analysis Results")
        
        # Check if data is ready
        if st.session_state.get('stores_ready') and st.session_state.get('users_ready'):
            if st.button("🧮 Calculate Distances", type="primary", use_container_width=True):
                with st.spinner("Analyzing distances..."):
                    results_df = calculate_distances(
                        st.session_state['stores_df'], 
                        st.session_state['users_df'], 
                        max_distance
                    )
                    st.session_state['results_df'] = results_df
        
        # Show results
        if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
            results_df = st.session_state['results_df']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Matches", len(results_df))
            with col2:
                st.metric("Eligible Users", results_df['user_idx'].nunique())
            with col3:
                st.metric("Avg Distance", f"{results_df['distance_km'].mean():.1f} km")
            with col4:
                st.metric("Min Distance", f"{results_df['distance_km'].min():.1f} km")
            
            # Eligible participants
            st.subheader("✅ Eligible Participants")
            eligible_users = results_df.loc[results_df.groupby('user_idx')['distance_km'].idxmin()]
            
            # Auto-detect important columns
            display_cols = ['user_idx', 'distance_km']
            user_cols = [c for c in results_df.columns if c.startswith('user_') and 
                        'latitude' not in c and 'longitude' not in c]
            display_cols.extend(user_cols[:4])  # Show first 4 user columns
            
            st.dataframe(
                eligible_users[display_cols].sort_values('distance_km'), 
                use_container_width=True
            )
            
            # Ineligible participants (too far away)
            if 'users_df' in st.session_state:
                all_user_indices = set(st.session_state['users_df'].index)
                eligible_indices = set(results_df['user_idx'].unique())
                ineligible_indices = all_user_indices - eligible_indices
                
                if ineligible_indices:
                    st.subheader("❌ Ineligible Participants (Too Far Away)")
                    ineligible_df = st.session_state['users_df'].loc[list(ineligible_indices)]
                    
                    # Add status column
                    ineligible_display = ineligible_df.copy()
                    ineligible_display['status'] = f'No stores within {max_distance}km'
                    
                    # Show relevant columns
                    show_cols = ['status']
                    for col in ineligible_df.columns[:4]:  # First 4 columns
                        if 'latitude' not in col and 'longitude' not in col:
                            show_cols.append(col)
                    
                    st.dataframe(ineligible_display[show_cols], use_container_width=True)
                    st.caption(f"💡 {len(ineligible_indices)} users have no nearby stores")
            
            # Detailed results
            st.subheader("🔍 Detailed Results")
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox("Sort by:", ["Distance", "User", "Store"])
                sort_mapping = {"Distance": "distance_km", "User": "user_idx", "Store": "store_idx"}
            
            with col2:
                max_show_dist = st.slider(
                    "Show pairs within distance (km):", 
                    0.0, 
                    float(results_df['distance_km'].max()), 
                    float(results_df['distance_km'].max())
                )
            
            # Apply filters
            filtered_results = results_df[results_df['distance_km'] <= max_show_dist]
            filtered_results = filtered_results.sort_values(sort_mapping[sort_by])
            
            st.dataframe(filtered_results, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Results (CSV)",
                data=csv,
                file_name=f"secret_shopper_results_{max_distance}km.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        else:
            st.info("👆 Upload and process both store and user data above, then calculate distances.")

# Initialize session state
if 'stores_ready' not in st.session_state:
    st.session_state['stores_ready'] = False
if 'users_ready' not in st.session_state:
    st.session_state['users_ready'] = False

if __name__ == "__main__":
    main()
