Test fixtures for laptop price model

Purpose
- Provide a small, deterministic dataset for fast, repeatable pytest runs without relying on the full production dataset laptop_price.csv.
- Exercise all branches of the project’s transformation code (weight parsing, screen feature extraction, storage parsing, CPU/RAM/OS grouping, etc.).

Schema
The CSV file fixtures/test_data.csv follows the exact column order and names of laptop_price.csv:
- Company
- TypeName
- Inches
- ScreenResolution
- Cpu
- Ram
- Memory
- Gpu
- OpSys
- Weight
- Price_euros

Encoding: UTF-8 (reader fixture falls back to latin-1 if needed).

Edge cases covered (at least two examples each)
Storage strings
- 256GB SSD (Rows: 1, 3, 10)
- 1TB HDD (Rows: 2, 8, 12, 16, 20 - includes 500GB HDD as additional HDD-only variety)
- 128GB SSD + 1TB HDD (Rows: 4, 11, 18)
- 2TB SSD (Rows: 7, 15)
- Flash Storage (Rows: 6, 13, 17 [32GB])

Screen strings
- 1366x768 (Rows: 3, 6, 10, 16, 20)
- 1920x1080 IPS (Rows: 12) and 1920x1080 non-IPS (Rows: 2, 15, 19)
- 3840x2160 Touchscreen (Rows: 5, 18)
- 3840x2160 IPS (Row: 9)
- 2560x1440 IPS Touchscreen (Rows: 11, 14)

Weight extremes
- 0.9kg (Row: 13)
- 3.5kg (Row: 9)

CPU varieties
- Intel Core i3 7100U 2.4GHz (Rows: 3, 15)
- Intel Core i7 8750H 2.2GHz (Rows: 2, 11)
- AMD Ryzen 5 2500U (Rows: 6, 16)
- Intel Core i9 9980HK (Rows: 7, 19)

GPU varieties
- Intel HD Graphics 620 (Rows: 1, 3, 4, 8, 15)
- Nvidia GeForce GTX 1050 (Rows: 2, 5, 9, 11, 18)
- AMD Radeon R5 (Rows: 6, 12, 16)

RAM sizes
- 4GB (Rows: 10, 12, 17, 20)
- 8GB (Rows: 1, 3, 4, 6, 8, 14)
- 16GB (Rows: 5, 11, 18)
- 32GB (Rows: 7, 9, 15, 19)

Operating systems
- Windows 10 (many rows)
- macOS (Rows: 1, 7, 13)
- Linux (Rows: 6, 16)
- No OS (Rows: 8, 20)
- Chrome OS (Rows: 10, 17)

Company names (including consolidated variants)
- Apple, Dell, HP, Lenovo, Asus, Acer, HP Compaq

Row-by-row notes (purpose)
1. Apple ultrabook, Retina-like IPS, SSD-only baseline.
2. Dell gaming, FHD, HDD-only, midweight, dedicated Nvidia GPU.
3. HP budget, HD screen, SSD-only, Core i3, Windows.
4. Lenovo convertible, FHD IPS touchscreen, hybrid storage (SSD + HDD).
5. Asus premium ultrabook, 4K touchscreen, SSD-only, dedicated Nvidia GPU.
6. Acer with Linux, HD screen, Flash storage, AMD CPU/GPU.
7. High-end Apple with Core i9 and 2TB SSD, premium pricing.
8. HP Compaq with consolidated company naming, HDD-only, No OS.
9. Heavy 17.3" workstation, 4K IPS, hybrid storage, 3.5kg.
10. Small netbook with HD screen, Core i3, light weight, Chrome OS.
11. Asus gaming with 1440p IPS touchscreen and hybrid storage.
12. Acer budget with FHD IPS, HDD-only, AMD GPU.
13. Ultra-light Apple with Flash storage at 0.9kg.
14. HP convertible with 1440p IPS touchscreen, SSD-only.
15. Dell with 2TB SSD extreme storage, Core i3, realistic high price.
16. Lenovo with AMD Ryzen and HDD-only, Linux.
17. Asus Chromebook with 32GB Flash, Chrome OS, low price.
18. Acer with 4K touchscreen, hybrid storage, Nvidia GPU.
19. HP with Core i9 and 1TB SSD, UHD iGPU, Windows 10.
20. Lenovo with Pentium, HDD-only, No OS.

Maintenance
- Keep exactly 20 rows to ensure fast tests and stable expectations in unit/integration tests.
- Ensure any changes continue to cover all edge cases above with at least two examples each.
- Validate that values still parse with the project’s transformation functions in Laptop Price model(1).py:
  - Weight parsing accepts kg and numeric values.
  - ScreenResolution strings include or omit keywords Touchscreen and IPS to exercise both branches.
  - Memory strings include SSD/HDD/Flash and combinations to validate extract_storage_features.
- When adding a new edge case, replace an existing redundant row rather than increasing the row count.
- Encoding should remain UTF-8; if special characters are introduced, tests can still read via latin-1 fallback.

How to update
- Edit fixtures/test_data.csv and update the Row-by-row notes above to reflect intent.
- Run tests: a smoke test could assert len(test_data) == 20 and validate key parsing.
