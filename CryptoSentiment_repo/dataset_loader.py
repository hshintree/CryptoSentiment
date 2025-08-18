import glob
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union, TYPE_CHECKING

import pandas as pd
import yaml

from tqdm import tqdm

if TYPE_CHECKING:
    from dataset_loader import DatasetLoader

def _discover_label_cols(columns: Iterable[str]) -> List[str]:
    """Find all columns that look like labels."""
    return [c for c in columns if c.lower() in ("label", "sentiment", "class")]

def _find_close_col(df: pd.DataFrame) -> str:
    """Return the name of the column that stores the daily *close* price."""
    for candidate in ("Close_x", "close", "Close", "closing"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not identify close price column (expected one of Close_x/close/Close/closing)")

class DatasetLoader:
    def __init__(
        self,
        config_path_or_loader: Union[str, "DatasetLoader"] = "config.yaml",
        *,
        include_labels: bool = True,
        label_reducer: Optional[Callable[[pd.Series], Union[float, int, str]]] = None,
    ):
        print(f"DatasetLoader.__init__ called with: {config_path_or_loader}")
        
        if isinstance(config_path_or_loader, DatasetLoader):
            # Copy attributes from existing loader
            print("Copying attributes from existing DatasetLoader")
            self.prebit_dir = config_path_or_loader.prebit_dir
            self.price_path = config_path_or_loader.price_path
            self.kaggle_dir = config_path_or_loader.kaggle_dir
            self.events_path = config_path_or_loader.events_path
        else:
            # Load from config file
            print(f"Loading config from: {config_path_or_loader}")
            with open(config_path_or_loader, "r") as f:
                cfg = yaml.safe_load(f)
            d = cfg["data"]

            # PreBit directories / single file
            self.prebit_dir    = Path(d["prebit_dir"])
            self.price_path    = Path(d["price_label_path"])
            self.kaggle_dir    = Path(d["kaggle_dir"])
            self.events_path   = Path(d["bitcoin_events_path"])

        self.include_labels = include_labels
        self.label_reducer  = label_reducer or (lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        
        print(f"DatasetLoader initialized with paths:")
        print(f"  prebit_dir: {self.prebit_dir}")
        print(f"  price_path: {self.price_path}")
        print(f"  kaggle_dir: {self.kaggle_dir}")
        print(f"  events_path: {self.events_path}")

    def load_dataset(self, *, aggregate: bool = False, save_to_csv: bool = True) -> pd.DataFrame:
        print("Starting dataset loading process...")
        
        # -- 1) Load PreBit tweets (2015-2021) --
        prebit_df = None
        if self.prebit_dir:
            print("Loading PreBit dataset...")
            prebit_df = self._load_prebit_dir()
            prebit_df = prebit_df.rename(columns={"Tweet Date": "date"})
            # Flag tweets from PreBit dataset
            prebit_df["data_source"] = "prebit"
            # Set metadata columns to NaN for PreBit tweets
            for col in ["hashtags", "is_retweet", "user_verified", "user_followers"]:
                prebit_df[col] = pd.NA
            print(f"Loaded PreBit dataset: {len(prebit_df)} rows")

        # -- 2) Load Kaggle tweets (2021-2023) with rich metadata --
        kaggle_df = None
        if self.kaggle_dir:
            print("Loading Kaggle dataset...")
            kaggle_df = self._load_kaggle_tweets()
            kaggle_df = kaggle_df.rename(columns={"Tweet Date": "date"})
            kaggle_df["data_source"] = "kaggle"
            # Ensure all metadata columns exist (even if not in source)
            for col in ["hashtags", "is_retweet", "user_verified", "user_followers"]:
                if col not in kaggle_df.columns:
                    kaggle_df[col] = pd.NA
            print(f"Loaded Kaggle dataset: {len(kaggle_df)} rows")

        # -- 3) Combine all tweets --
        print("Combining all tweet datasets...")
        parts = []
        if prebit_df is not None:
            parts.append(prebit_df)
        if kaggle_df is not None:
            parts.append(kaggle_df)
        df = pd.concat(parts, ignore_index=True).sort_values("date")
        print(f"Combined dataset: {len(df)} rows")

        # -- 4) Merge price data with ALL tweets --
        print("Loading and merging price data...")
        price_df = self._load_price_data()
        
        # Debug price data
        print(f"Price data date range: {price_df['date'].min()} to {price_df['date'].max()}")
        print(f"Price data shape: {price_df.shape}")
        print(f"Price data columns: {list(price_df.columns)}")
        print(f"Sample price data:")
        print(price_df.head())
        
        # Debug tweet data date range
        print(f"Tweet data date range: {df['date'].min()} to {df['date'].max()}")
        print(f"Tweet data shape: {df.shape}")
        print(f"Tweet data columns: {list(df.columns)}")
        
        # Check for date format mismatches
        print(f"Price date dtype: {price_df['date'].dtype}")
        print(f"Tweet date dtype: {df['date'].dtype}")
        
        # Debug: Check for suspicious dates
        print(f"Tweet date statistics:")
        print(f"  Earliest date: {df['date'].min()}")
        print(f"  Latest date: {df['date'].max()}")
        print(f"  Date range in days: {(df['date'].max() - df['date'].min()).days}")
        print(f"  Unique dates: {df['date'].nunique()}")
        
        # Check for dates that seem wrong (before 2010)
        suspicious_dates = df[df['date'].dt.year < 2010]
        if len(suspicious_dates) > 0:
            print(f"⚠️  WARNING: Found {len(suspicious_dates)} tweets with suspicious dates (before 2010)")
            print(f"  Sample suspicious dates: {suspicious_dates['date'].head(5).tolist()}")
            print(f"  Sample suspicious tweets:")
            for i, row in suspicious_dates.head(3).iterrows():
                print(f"    {row['date']}: {row['Tweet Content'][:100]}...")
        
        # Check coverage of tweet dates in price data
        tweet_dates = set(df['date'].dt.date)
        price_dates = set(price_df['date'].dt.date)
        missing_dates = tweet_dates - price_dates
        
        if missing_dates:
            print(f"WARNING: {len(missing_dates)} tweet dates missing from price data")
            print(f"Missing dates: {sorted(missing_dates)[:5]}...")  # Show first 5 missing dates
            
            # Create complete price data for missing dates using forward/backward fill
            print("Filling missing price data for tweet dates...")
            all_tweet_dates = pd.to_datetime(sorted(tweet_dates))
            complete_price_df = pd.DataFrame({'date': all_tweet_dates})
            complete_price_df = complete_price_df.merge(price_df, on='date', how='left')
            complete_price_df['Close'] = complete_price_df['Close'].ffill().bfill()  # Fixed deprecated method
            complete_price_df['Volume'] = complete_price_df['Volume'].fillna(0)
        else:
            print("✓ All tweet dates have corresponding price data!")
            complete_price_df = price_df
        
        # Debug the complete price data before merge
        print(f"Complete price data shape: {complete_price_df.shape}")
        print(f"Complete price data columns: {list(complete_price_df.columns)}")
        print(f"Sample complete price data:")
        print(complete_price_df.head())
        
        # Merge with tweets - use date only for better matching
        before_merge = len(df)
        print(f"Before merge - df columns: {list(df.columns)}")
        print(f"Before merge - complete_price_df columns: {list(complete_price_df.columns)}")
        
        # Handle column name conflicts - if tweets already have Close column, we need to handle this
        if 'Close' in df.columns:
            print("Found existing Close column in tweets, handling merge conflict...")
            # Rename the price data Close column to avoid conflict
            complete_price_df = complete_price_df.rename(columns={'Close': 'Close_price'})
            print(f"Renamed price Close to Close_price")
        
        # Convert tweet dates to date only for better matching with price data
        df_merge = df.copy()
        df_merge['date_only'] = df_merge['date'].dt.date
        complete_price_df['date_only'] = complete_price_df['date'].dt.date
        
        # Merge on date_only for better price coverage
        df_merged = df_merge.merge(complete_price_df, on='date_only', how='left', suffixes=('', '_price'))
        
        # Restore original date column and drop temporary date_only
        df_merged = df_merged.drop(columns=['date_only'])
        
        after_merge = len(df_merged)
        
        print(f"After merge - df columns: {list(df_merged.columns)}")
        print(f"Before price merge: {before_merge} rows")
        print(f"After price merge: {after_merge} rows")
        
        # Handle the Close column after merge
        if 'Close_price' in df_merged.columns:
            # If we had a conflict, use the price data Close and drop the old one
            if 'Close' in df_merged.columns:
                print("Replacing existing Close column with price data Close")
                df_merged = df_merged.drop(columns=['Close'])
            df_merged = df_merged.rename(columns={'Close_price': 'Close'})
            print("Renamed Close_price back to Close")
        
        # Check if Close column exists after merge
        if 'Close' not in df_merged.columns:
            print("ERROR: Close column not found after merge!")
            print(f"Available columns: {list(df_merged.columns)}")
            raise ValueError("Close column missing after price merge")
        
        print(f"Rows with price data: {df_merged['Close'].notna().sum()} ({100 * df_merged['Close'].notna().mean():.2f}%)")
        
        # Verify all rows now have price data
        missing_prices = df_merged['Close'].isna().sum()
        if missing_prices > 0:
            print(f"WARNING: {missing_prices} rows still missing price data!")
            # Show examples of missing prices
            no_price = df_merged[df_merged['Close'].isna()].head(3)
            print(f"Sample tweets without price data:")
            for i, row in no_price.iterrows():
                print(f"  {row['date']}: {row['Tweet Content'][:100]}...")
        else:
            print("✓ All tweets now have price data!")
        
        # Show some examples of tweets with price data
        with_price = df_merged[df_merged['Close'].notna()].head(3)
        if len(with_price) > 0:
            print(f"Sample tweets with price data:")
            for i, row in with_price.iterrows():
                print(f"  {row['date']}: ${row['Close']:.2f} - {row['Tweet Content'][:100]}...")

        # -- 5) Add metadata availability flags --
        df_merged["has_user_metadata"] = df_merged["user_followers"].notna()
        df_merged["has_tweet_metadata"] = df_merged["hashtags"].notna()

        # -- 6) Merge in events as a one-hot Is_Event flag --
        if self.events_path:
            print("Loading and merging events data...")
            events_df = self._load_events_data()
            df_merged["Is_Event"] = df_merged["date"].isin(events_df["date"]).astype(int)
        else:
            df_merged["Is_Event"] = 0

        # -- 7) Per-day aggregation if requested --
        if aggregate:
            print("Performing daily aggregation...")
            df_merged = self._aggregate_per_day(df_merged)
            print(f"After aggregation: {len(df_merged)} rows")

        # -- 8) Save to CSV if requested --
        if save_to_csv:
            self._save_dataset_to_csv(df_merged, aggregate)
        
        print("Dataset loading complete!")
        return df_merged

    def _save_dataset_to_csv(self, df: pd.DataFrame, aggregate: bool = False):
        """Save the final dataset to a CSV file in a data folder."""
        print("Saving dataset to CSV...")
        
        # Create data directory if it doesn't exist
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Generate filename based on aggregation
        if aggregate:
            filename = "combined_dataset_aggregated.csv"
        else:
            filename = "combined_dataset_raw.csv"
        
        filepath = data_dir / filename
        
        print(f"Saving {len(df)} rows to {filepath}")
        print(f"Dataset size: {df.memory_usage(deep=True).sum() / 1024**3:.2f} GB")
        
        # Save with progress tracking
        try:
            # Use chunksize for large datasets to avoid memory issues
            if len(df) > 1000000:  # If more than 1M rows, save in chunks
                print("Large dataset detected, saving in chunks...")
                chunk_size = 100000
                chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
                
                # Write header for first chunk
                chunks[0].to_csv(filepath, index=False, mode='w')
                print(f"Wrote header and first chunk ({len(chunks[0])} rows)")
                
                # Append remaining chunks
                for i, chunk in enumerate(chunks[1:], 1):
                    chunk.to_csv(filepath, index=False, mode='a', header=False)
                    print(f"Wrote chunk {i+1}/{len(chunks)} ({len(chunk)} rows)")
            else:
                # For smaller datasets, save directly
                df.to_csv(filepath, index=False)
                print("Dataset saved directly")
            
            print(f"Dataset successfully saved to {filepath}")
            print(f"File size: {filepath.stat().st_size / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"Error saving dataset: {e}")
            print("Attempting to save with reduced precision...")
            try:
                # Try saving with reduced precision to save space
                df.to_csv(filepath, index=False, float_format='%.6f')
                print(f"Dataset saved with reduced precision to {filepath}")
            except Exception as e2:
                print(f"Failed to save dataset: {e2}")
                print("Dataset will not be saved to CSV")

    def _load_kaggle_tweets(self) -> pd.DataFrame:
        """Load the additional Kaggle tweets from directory—must have date + text col."""
        print("Starting Kaggle tweets loading...")
        dir_path = Path(self.kaggle_dir)
        if not dir_path.exists():
            raise FileNotFoundError(dir_path)

        # Load both Kaggle tweet files
        first_file = dir_path / "first_kaggle_tweets.csv"
        second_file = dir_path / "second_kaggle_tweets.csv"
        
        if not first_file.exists():
            raise FileNotFoundError(f"First Kaggle tweets file not found: {first_file}")
        if not second_file.exists():
            raise FileNotFoundError(f"Second Kaggle tweets file not found: {second_file}")

        print(f"Loading first file: {first_file}")
        print(f"Loading second file: {second_file}")

        # Follower threshold for filtering
        min_followers = 2000
        print(f"Filtering tweets by user followers >= {min_followers}")

        # Load files in chunks to be more memory-efficient
        def load_csv_in_chunks(file_path, chunk_size=10000):
            """Load CSV file in chunks to avoid memory issues."""
            print(f"Loading {file_path} in chunks of {chunk_size}...")
            
            # First, try to read just the header to get column names
            try:
                header_df = pd.read_csv(file_path, nrows=0, encoding='utf-8')
            except UnicodeDecodeError:
                header_df = pd.read_csv(file_path, nrows=0, encoding='latin-1')
            
            print(f"Columns found: {list(header_df.columns)}")
            
            # Find required columns
            date_cols = [c for c in header_df.columns if c.lower() in ("date", "tweet date")]
            text_cols = [c for c in header_df.columns if c.lower() in ("text", "tweet", "text_split")]
            follower_cols = [c for c in header_df.columns if c.lower() in ("user_followers", "followers", "follower_count")]
            
            if not date_cols:
                raise ValueError(f"No date column found in {file_path}")
            if not text_cols:
                raise ValueError(f"No text column found in {file_path}")
            if not follower_cols:
                print(f"Warning: No user_followers column found in {file_path}, will keep all tweets")
                follower_col = None
            else:
                follower_col = follower_cols[0]
            
            date_col = date_cols[0]
            text_col = text_cols[0]
            
            print(f"Using date column: {date_col}")
            print(f"Using text column: {text_col}")
            if follower_col:
                print(f"Using follower column: {follower_col}")
            
            # Count total rows for progress bar (approximate)
            try:
                total_rows = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1  # -1 for header
            except UnicodeDecodeError:
                total_rows = sum(1 for _ in open(file_path, 'r', encoding='latin-1')) - 1
            
            print(f"Total rows in {file_path.name}: {total_rows}")
            
            # Load in chunks with error handling
            chunks = []
            chunk_count = 0
            processed_rows = 0
            filtered_rows = 0
            skipped_chunks = 0
            
            with tqdm(total=total_rows, desc=f"Loading {file_path.name}") as pbar:
                # Try UTF-8 first
                try:
                    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8', low_memory=False, on_bad_lines='skip')
                except UnicodeDecodeError:
                    # Fall back to latin-1
                    chunk_reader = pd.read_csv(file_path, chunksize=chunk_size, encoding='latin-1', low_memory=False, on_bad_lines='skip')
                
                for chunk in chunk_reader:
                    chunk_count += 1
                    
                    try:
                        # Keep needed columns
                        keep_cols = [date_col, text_col]
                        if follower_col:
                            keep_cols.append(follower_col)
                        
                        chunk = chunk[keep_cols].copy()
                        
                        # Convert date column
                        chunk[date_col] = pd.to_datetime(chunk[date_col], errors='coerce')
                        
                        # Debug: Check for suspicious dates in this chunk
                        if len(chunk) > 0:
                            suspicious_in_chunk = chunk[chunk[date_col].dt.year < 2010]
                            if len(suspicious_in_chunk) > 0:
                                print(f"⚠️  Found {len(suspicious_in_chunk)} suspicious dates in chunk {chunk_count}")
                                print(f"  Sample: {suspicious_in_chunk[date_col].head(3).tolist()}")
                        
                        # Drop rows with invalid dates
                        chunk = chunk.dropna(subset=[date_col])
                        
                        # Filter by user followers if column exists
                        if follower_col and follower_col in chunk.columns:
                            # Convert to numeric, handling any non-numeric values
                            chunk[follower_col] = pd.to_numeric(chunk[follower_col], errors='coerce')
                            
                            # Filter by follower count
                            before_filter = len(chunk)
                            chunk = chunk[chunk[follower_col] >= min_followers]
                            after_filter = len(chunk)
                            filtered_rows += (before_filter - after_filter)
                        
                        if len(chunk) > 0:
                            chunks.append(chunk)
                            processed_rows += len(chunk)
                            pbar.update(len(chunk))
                        
                        # Print progress every 50 chunks
                        if chunk_count % 50 == 0:
                            print(f"Processed {chunk_count} chunks, {processed_rows} valid rows, {filtered_rows} filtered rows, {skipped_chunks} skipped chunks")
                            
                    except Exception as e:
                        print(f"Error processing chunk {chunk_count}: {e}")
                        skipped_chunks += 1
                        # Continue with next chunk instead of failing completely
                        continue
            
            print(f"Loaded {len(chunks)} chunks from {file_path.name}")
            print(f"Total processed rows: {processed_rows}")
            print(f"Total filtered rows (followers < {min_followers}): {filtered_rows}")
            print(f"Skipped chunks due to errors: {skipped_chunks}")
            
            # Combine chunks
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                print(f"Combined chunks: {len(result)} rows")
                return result, date_col, text_col, follower_col
            else:
                raise ValueError(f"No valid data found in {file_path}")

        def load_csv_robust(file_path):
            """Alternative robust loading method for severely corrupted files."""
            print(f"Attempting robust loading for {file_path}...")
            
            # Try different parsing strategies (removed deprecated error_bad_lines)
            strategies = [
                {'encoding': 'utf-8', 'on_bad_lines': 'skip'},
                {'encoding': 'latin-1', 'on_bad_lines': 'skip'},
                {'encoding': 'utf-8', 'on_bad_lines': 'warn'},
                {'encoding': 'latin-1', 'on_bad_lines': 'warn'},
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    print(f"Trying strategy {i+1}: {strategy}")
                    df = pd.read_csv(file_path, **strategy)
                    
                    # Find columns
                    date_cols = [c for c in df.columns if c.lower() in ("date", "tweet date")]
                    text_cols = [c for c in df.columns if c.lower() in ("text", "tweet", "text_split")]
                    follower_cols = [c for c in df.columns if c.lower() in ("user_followers", "followers", "follower_count")]
                    
                    if date_cols and text_cols:
                        date_col = date_cols[0]
                        text_col = text_cols[0]
                        follower_col = follower_cols[0] if follower_cols else None
                        
                        # Keep needed columns
                        keep_cols = [date_col, text_col]
                        if follower_col:
                            keep_cols.append(follower_col)
                        
                        df = df[keep_cols].copy()
                        
                        # Convert date column
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        
                        # Drop rows with invalid dates
                        df = df.dropna(subset=[date_col])
                        
                        # Filter by user followers if column exists
                        if follower_col and follower_col in df.columns:
                            df[follower_col] = pd.to_numeric(df[follower_col], errors='coerce')
                            before_filter = len(df)
                            df = df[df[follower_col] >= min_followers]
                            after_filter = len(df)
                            print(f"Filtered {before_filter - after_filter} rows by follower count")
                        
                        print(f"Successfully loaded {len(df)} rows with strategy {i+1}")
                        return df, date_col, text_col, follower_col
                        
                except Exception as e:
                    print(f"Strategy {i+1} failed: {e}")
                    continue
            
            raise ValueError(f"All loading strategies failed for {file_path}")

        def load_csv_line_by_line(file_path):
            """Load CSV file line by line to handle severe corruption."""
            print(f"Attempting line-by-line loading for {file_path}...")
            
            # First, read header to get column names
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    header = f.readline().strip()
            except UnicodeDecodeError:
                with open(file_path, 'r', encoding='latin-1') as f:
                    header = f.readline().strip()
            
            columns = header.split(',')
            print(f"Columns found: {columns}")
            
            # Find required columns
            date_cols = [c for c in columns if c.lower() in ("date", "tweet date")]
            text_cols = [c for c in columns if c.lower() in ("text", "tweet", "text_split")]
            follower_cols = [c for c in columns if c.lower() in ("user_followers", "followers", "follower_count")]
            
            if not date_cols or not text_cols:
                raise ValueError(f"Required columns not found in {file_path}")
            
            date_idx = columns.index(date_cols[0])
            text_idx = columns.index(text_cols[0])
            follower_idx = columns.index(follower_cols[0]) if follower_cols else None
            
            # Read file line by line
            rows = []
            line_count = 0
            valid_rows = 0
            filtered_rows = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    next(f)  # Skip header
                    for line in f:
                        line_count += 1
                        if line_count % 100000 == 0:
                            print(f"Processed {line_count} lines, {valid_rows} valid rows, {filtered_rows} filtered rows")
                        
                        try:
                            # Split by comma, but handle quoted fields
                            parts = line.strip().split(',')
                            if len(parts) > max(date_idx, text_idx):
                                date_val = parts[date_idx].strip('"')
                                text_val = parts[text_idx].strip('"')
                                
                                # Check follower count if available
                                if follower_idx is not None and len(parts) > follower_idx:
                                    try:
                                        follower_val = float(parts[follower_idx].strip('"'))
                                        if follower_val < min_followers:
                                            filtered_rows += 1
                                            continue
                                    except (ValueError, IndexError):
                                        # If can't parse followers, skip this row
                                        continue
                                
                                # Try to parse date
                                try:
                                    parsed_date = pd.to_datetime(date_val)
                                    if pd.notna(parsed_date):
                                        row_data = [parsed_date, text_val]
                                        if follower_idx is not None and len(parts) > follower_idx:
                                            row_data.append(float(parts[follower_idx].strip('"')))
                                        rows.append(row_data)
                                        valid_rows += 1
                                except:
                                    continue  # Skip invalid dates
                        except:
                            continue  # Skip malformed lines
                            
            except UnicodeDecodeError:
                # Retry with latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    next(f)  # Skip header
                    for line in f:
                        line_count += 1
                        if line_count % 100000 == 0:
                            print(f"Processed {line_count} lines, {valid_rows} valid rows, {filtered_rows} filtered rows")
                        
                        try:
                            parts = line.strip().split(',')
                            if len(parts) > max(date_idx, text_idx):
                                date_val = parts[date_idx].strip('"')
                                text_val = parts[text_idx].strip('"')
                                
                                # Check follower count if available
                                if follower_idx is not None and len(parts) > follower_idx:
                                    try:
                                        follower_val = float(parts[follower_idx].strip('"'))
                                        if follower_val < min_followers:
                                            filtered_rows += 1
                                            continue
                                    except (ValueError, IndexError):
                                        continue
                                
                                try:
                                    parsed_date = pd.to_datetime(date_val)
                                    if pd.notna(parsed_date):
                                        row_data = [parsed_date, text_val]
                                        if follower_idx is not None and len(parts) > follower_idx:
                                            row_data.append(float(parts[follower_idx].strip('"')))
                                        rows.append(row_data)
                                        valid_rows += 1
                                except:
                                    continue
                        except:
                            continue
            
            print(f"Line-by-line loading complete: {valid_rows} valid rows, {filtered_rows} filtered rows from {line_count} lines")
            
            if rows:
                columns_to_use = [date_cols[0], text_cols[0]]
                if follower_cols:
                    columns_to_use.append(follower_cols[0])
                df = pd.DataFrame(rows, columns=columns_to_use)
                return df, date_cols[0], text_cols[0], follower_cols[0] if follower_cols else None
            else:
                raise ValueError(f"No valid data found in {file_path}")

        # Load both files
        print("Loading first Kaggle file...")
        try:
            first_df, first_date_col, first_text_col, first_follower_col = load_csv_in_chunks(first_file)
        except Exception as e:
            print(f"Chunked loading failed for first file: {e}")
            print("Falling back to robust loading method...")
            try:
                first_df, first_date_col, first_text_col, first_follower_col = load_csv_robust(first_file)
            except Exception as e2:
                print(f"Robust loading failed for first file: {e2}")
                print("Falling back to line-by-line loading method...")
                first_df, first_date_col, first_text_col, first_follower_col = load_csv_line_by_line(first_file)
        
        print("Loading second Kaggle file...")
        try:
            second_df, second_date_col, second_text_col, second_follower_col = load_csv_in_chunks(second_file)
        except Exception as e:
            print(f"Chunked loading failed for second file: {e}")
            print("Falling back to robust loading method...")
            try:
                second_df, second_date_col, second_text_col, second_follower_col = load_csv_robust(second_file)
            except Exception as e2:
                print(f"Robust loading failed for second file: {e2}")
                print("Falling back to line-by-line loading method...")
                second_df, second_date_col, second_text_col, second_follower_col = load_csv_line_by_line(second_file)

        # Combine both DataFrames
        print("Combining both Kaggle files...")
        df = pd.concat([first_df, second_df], ignore_index=True)
        print(f"Combined dataset: {len(df)} rows")

        # Standardize column names
        df = df.rename(columns={first_date_col: "Tweet Date", first_text_col: "Tweet Content"})
        if first_follower_col:
            df = df.rename(columns={first_follower_col: "user_followers"})
        
        # if labels present, keep them (though we're not loading them in chunks for now)
        if not self.include_labels:
            df = df.drop(columns=_discover_label_cols(df.columns), errors="ignore")

        print(f"Final Kaggle dataset: {len(df)} rows")
        
        # Return columns including user_followers if available
        return_cols = ["Tweet Date", "Tweet Content"]
        if "user_followers" in df.columns:
            return_cols.append("user_followers")
        return_cols.extend(_discover_label_cols(df.columns))
        
        return df[return_cols]

    # -- reuse existing _load_single_prebit, _load_prebit_dir, _aggregate_per_day --  

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single_prebit(self) -> pd.DataFrame:
        """Load the (older) *single‑CSV* version of the PreBit dataset."""
        csv_path = Path(self.prebit_dir)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path, parse_dates=["date"])  # assumed column names
        required = {"date", "tweet", "close"}
        if not required.issubset(df.columns):
            raise ValueError(
                f"Expected at least columns {required}, got {set(df.columns)}"
            )

        if not self.include_labels:
            df = df.drop(columns=_discover_label_cols(df.columns), errors="ignore")

        return df.rename(columns={"date": "Tweet Date", "tweet": "Tweet Content", "close": "Close"})

    def _load_prebit_dir(self) -> pd.DataFrame:
        """Load the *directory* layout shipped on Kaggle (6 tweet CSVs + price)."""
        print("Starting PreBit directory loading...")
        dir_path = Path(self.prebit_dir)
        if not dir_path.exists():
            raise FileNotFoundError(dir_path)

        # ---- 1. Tweets ---------------------------------------------------
        print("Looking for tweet CSV files...")
        tweet_csvs = sorted(dir_path.glob("combined_tweets_*_labeled.csv"))
        if not tweet_csvs:
            raise FileNotFoundError("No tweet CSVs matching 'combined_tweets_*_labeled.csv' found in " + str(dir_path))

        print(f"Found {len(tweet_csvs)} tweet CSV files: {[f.name for f in tweet_csvs]}")

        tweet_frames: List[pd.DataFrame] = []
        for i, csv in enumerate(tqdm(tweet_csvs, desc="Loading PreBit tweet files")):
            print(f"Loading tweet file {i+1}/{len(tweet_csvs)}: {csv.name}")
            df = pd.read_csv(csv, parse_dates=["date"])
            required = {"date", "text_split"}
            if not required.issubset(df.columns):
                raise ValueError(f"{csv} is missing one of {required}")

            keep_cols = ["date", "text_split"]
            if self.include_labels:
                label_cols = _discover_label_cols(df.columns)
                keep_cols.extend(label_cols)
            tweet_frames.append(df[keep_cols])
            print(f"Loaded {len(df)} rows from {csv.name}")

        print("Combining all PreBit tweet files...")
        tweets = pd.concat(tweet_frames, ignore_index=True)
        tweets = tweets.rename(columns={"date": "Tweet Date", "text_split": "Tweet Content"})
        print(f"Combined PreBit tweets: {len(tweets)} rows")

        # ---- 2. Price ----------------------------------------------------
        print("Looking for price data...")
        price_path = dir_path / "old_price_label.csv"
        if not price_path.exists():
            # price file optional – we simply return tweets only
            print("No price file found, returning tweets only")
            return tweets

        print(f"Loading price data from {price_path}")
        price_df = pd.read_csv(price_path, parse_dates=["date"])
        close_col = _find_close_col(price_df)

        price_keep = ["date", close_col]
        if self.include_labels:
            price_keep.extend(_discover_label_cols(price_df.columns))

        price_df = price_df[price_keep]
        price_df = price_df.rename(columns={"date": "Tweet Date", close_col: "Close"})
        print(f"Loaded price data: {len(price_df)} rows")

        # ---- 3. Merge ----------------------------------------------------
        # Many tweets → one price row per day ⇒ left join keeps all tweets.
        print("Merging tweets with price data...")
        merged = tweets.merge(price_df, on="Tweet Date", how="left", suffixes=("", "_price"))
        print(f"Final PreBit dataset: {len(merged)} rows")
        return merged

    # ------------------------------------------------------------------
    # Aggregation helper
    # ------------------------------------------------------------------

    def _aggregate_per_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """Collapse multiple tweets into a single record per date."""
        print(f"Starting aggregation of {len(df)} rows...")
        print(f"Date range before aggregation: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique dates before aggregation: {df['date'].nunique()}")
        
        # Check for suspicious dates before aggregation
        suspicious_before = df[df['date'].dt.year < 2010]
        if len(suspicious_before) > 0:
            print(f"⚠️  Found {len(suspicious_before)} rows with suspicious dates before aggregation")
            print(f"  Sample suspicious dates: {suspicious_before['date'].head(5).tolist()}")
        
        # Convert datetime to date for proper daily aggregation
        df_agg = df.copy()
        df_agg['date_only'] = df_agg['date'].dt.date
        print(f"Converting datetime to date for aggregation...")
        print(f"Unique dates after conversion: {df_agg['date_only'].nunique()}")
        
        label_cols = _discover_label_cols(df.columns) if self.include_labels else []

        numeric_cols = df_agg.select_dtypes("number").columns.difference(label_cols)
        string_cols = df_agg.select_dtypes("object").columns.difference(["Tweet Content"])

        def _agg_fn(series: pd.Series):
            if series.name in label_cols:
                return self.label_reducer(series)
            if series.name in numeric_cols:
                # For price data, use the first non-null value (should be same for all rows per date)
                if series.name in ['Close', 'Volume']:
                    return series.dropna().iloc[0] if not series.dropna().empty else series.iloc[0]
                return series.mean()
            if series.name == "Tweet Content":
                return " \n".join(series.astype(str))
            # fallback
            return series.iloc[0]

        # Group by date_only (not datetime) for proper daily aggregation
        grouped = df_agg.groupby("date_only", as_index=False).agg(_agg_fn)
        
        # Convert date_only back to datetime for consistency
        grouped['date'] = pd.to_datetime(grouped['date_only'])
        grouped = grouped.drop(columns=['date_only'])
        
        print(f"Aggregation complete: {len(grouped)} unique dates")
        print(f"Date range after aggregation: {grouped['date'].min()} to {grouped['date'].max()}")
        
        # Check for suspicious dates after aggregation
        suspicious_after = grouped[grouped['date'].dt.year < 2010]
        if len(suspicious_after) > 0:
            print(f"⚠️  Found {len(suspicious_after)} rows with suspicious dates after aggregation")
            print(f"  Sample suspicious dates: {suspicious_after['date'].head(5).tolist()}")
        
        # Verify that price data is preserved after aggregation
        print(f"Rows with Close price after aggregation: {grouped['Close'].notna().sum()} ({100 * grouped['Close'].notna().mean():.2f}%)")
        
        # Check for any missing prices after aggregation
        missing_after_agg = grouped['Close'].isna().sum()
        if missing_after_agg > 0:
            print(f"WARNING: {missing_after_agg} dates missing Close price after aggregation!")
            # Try to fill any remaining missing prices
            grouped['Close'] = grouped['Close'].ffill().bfill()  # Fixed deprecated method
            print(f"After filling: {grouped['Close'].notna().sum()} dates have Close price")
        
        return grouped

    def _load_price_data(self) -> pd.DataFrame:
        """
        Load daily close price data, combining old PreBit data (2015-2017) with new data (2018-2025).
        Returns a DataFrame with 'date', 'Close', and 'Volume' columns.
        """
        print(f"Loading new price data from {self.price_path}")
        new_price_df = pd.read_csv(self.price_path, parse_dates=["Open time"])
        # Rename columns for consistency
        new_price_df = new_price_df.rename(columns={"Open time": "date", "Close": "Close", "Volume": "Volume"})
        # Keep date, Close and Volume columns
        new_price_df = new_price_df[["date", "Close", "Volume"]]
        print(f"Loaded new price data: {len(new_price_df)} rows ({new_price_df['date'].min()} to {new_price_df['date'].max()})")
        
        # Load old price data from PreBit directory if available
        old_price_path = Path(self.prebit_dir) / "old_price_label.csv"
        if old_price_path.exists():
            print(f"Loading old price data from {old_price_path}")
            old_price_df = pd.read_csv(old_price_path, parse_dates=["date"])
            
            # Find close column in old data (should be Close_x)
            close_col = _find_close_col(old_price_df)
            old_price_df = old_price_df.rename(columns={close_col: "Close"})
            
            # Check if Volume column exists
            if "Volume" in old_price_df.columns:
                print("Found Volume column in old price data")
            else:
                print("No Volume column found, adding Volume=0")
                old_price_df["Volume"] = 0
            
            # Keep only needed columns
            old_price_df = old_price_df[["date", "Close", "Volume"]]
            print(f"Loaded old price data: {len(old_price_df)} rows ({old_price_df['date'].min()} to {old_price_df['date'].max()})")
            
            # Combine old and new price data
            print("Combining old and new price data...")
            combined_price_df = pd.concat([old_price_df, new_price_df], ignore_index=True)
            
            # Remove duplicates (in case of overlap) and sort by date
            combined_price_df = combined_price_df.drop_duplicates(subset=['date']).sort_values('date')
            
            print(f"Combined price data: {len(combined_price_df)} rows ({combined_price_df['date'].min()} to {combined_price_df['date'].max()})")
            
            # Check for any gaps in the data
            date_range = pd.date_range(combined_price_df['date'].min(), combined_price_df['date'].max(), freq='D')
            missing_dates = date_range.difference(combined_price_df['date'])
            if len(missing_dates) > 0:
                print(f"Found {len(missing_dates)} missing dates in price data")
                print(f"Missing date range: {missing_dates.min()} to {missing_dates.max()}")
                
                # Fill missing dates with forward fill
                complete_dates = pd.DataFrame({'date': date_range})
                combined_price_df = complete_dates.merge(combined_price_df, on='date', how='left')
                combined_price_df['Close'] = combined_price_df['Close'].ffill().bfill()  # Fixed deprecated method
                combined_price_df['Volume'] = combined_price_df['Volume'].fillna(0)
                print(f"After filling gaps: {len(combined_price_df)} rows")
            
            return combined_price_df
        else:
            print("Old price data not found, using only new price data")
        
        return new_price_df

    def _load_events_data(self) -> pd.DataFrame:
        """
        Load the events CSV and return a DataFrame with a 'date' column.
        """
        print(f"Loading events data from {self.events_path}")
        
        # First, let's check what columns are actually in the file
        try:
            # Read just the header to see what columns exist
            header_df = pd.read_csv(self.events_path, nrows=0)
            
            # Find date columns - the actual file has 'date' column
            date_cols = [c for c in header_df.columns if c.lower() in ("date", "Date")]
            if not date_cols:
                print("Warning: No date column found in events file")
                return pd.DataFrame(columns=["date"])
            
            date_col = date_cols[0]
            print(f"Using date column: {date_col}")
            
            # Load the full file with the correct date column
            df = pd.read_csv(self.events_path, parse_dates=[date_col])
            
            # Standardize column name
            if date_col != "date":
                df = df.rename(columns={date_col: "date"})
            
            # For the bitcoin_event.csv file, we have additional columns
            # Let's also check if we have the price columns for additional analysis
            if 'bitcoin_price' in df.columns:
                print(f"Found bitcoin price data: {len(df)} events with price information")
                print(f"Price range: ${df['bitcoin_price'].min():.2f} - ${df['bitcoin_price'].max():.2f}")
            
            if 'title' in df.columns:
                print(f"Found event titles: {len(df)} events with titles")
                print("Sample event titles:")
                for i, title in enumerate(df['title'].head(3)):
                    print(f"  {i+1}. {title}")
            
            result = df[["date"]].drop_duplicates()
            print(f"Loaded events data: {len(result)} unique dates")
            return result
            
        except Exception as e:
            print(f"Error loading events data: {e}")
            print("Returning empty events DataFrame")
            return pd.DataFrame(columns=["date"])

    def load_saved_dataset(self, aggregate: bool = False) -> pd.DataFrame:
        """Load the previously saved dataset from CSV for faster testing."""
        print("Loading saved dataset from CSV...")
        
        data_dir = Path("data")
        if aggregate:
            filepath = data_dir / "combined_dataset_aggregated.csv"
        else:
            filepath = data_dir / "combined_dataset_raw.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Saved dataset not found: {filepath}. Run load_dataset() first.")
        
        print(f"Loading from {filepath}")
        df = pd.read_csv(filepath, parse_dates=["date"])
        print(f"Loaded {len(df)} rows from saved dataset")
        
        return df

    def aggregate_saved_dataset(self) -> pd.DataFrame:
        """Load the saved raw dataset and aggregate it to daily level."""
        print("Loading saved raw dataset and aggregating...")
        
        # Load the saved raw dataset
        df = self.load_saved_dataset(aggregate=False)
        
        # Ensure date column is datetime - handle mixed formats
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            print("Converting date column to datetime...")
            try:
                # Try parsing with mixed format to handle different date formats
                df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
                print("Date parsing with mixed format successful")
            except Exception as e:
                print(f"Mixed format parsing failed: {e}")
                # Try alternative approach
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                print("Date parsing with errors='coerce' successful")
        
        # Check for any failed date parsing
        failed_dates = df['date'].isna().sum()
        if failed_dates > 0:
            print(f"⚠️  {failed_dates} rows with failed date parsing - removing them")
            df = df.dropna(subset=['date'])
        
        # Perform aggregation
        print("Performing daily aggregation...")
        df_agg = self._aggregate_per_day(df)
        
        # Save the aggregated version
        self._save_dataset_to_csv(df_agg, aggregate=True)
        
        return df_agg

