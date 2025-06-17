# Task Completion Checklist

When completing any coding task in ScatterLabel, ensure:

1. **Code Quality**
   - Run mypy for type checking
   - All functions/classes have proper docstrings
   - Using absolute imports (not relative)
   - No system path modifications

2. **Testing**
   - Write tests for new functionality
   - Verify pre-commit hooks pass by running: `git add . && git commit -m "test"`

3. **Progress Tracking**
   - Add tqdm for any long-running operations
   - Provide meaningful progress descriptions

4. **Deprecation Handling**
   - Address any deprecation warnings immediately
   - Update to recommended alternatives

5. **Database**
   - Use psycopg (v3) for PostgreSQL connections
   - Use POSTGRES_URI environment variable

6. **CMR Dataset Testing**
   When testing instance segmentation:
   - Train: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:23.077836_train_ordered.json`
   - Val: `/home/georgepearse/data/cmr/annotations/2025-05-15_12:38:38.270134_val_ordered.json`
   - Images: `/home/georgepearse/data/images/`
   - Always verify number of classes matches model architecture

7. **Full Implementation**
   - Never leave simplified placeholders
   - Complete all TODOs before ending session
   - Implement full solutions, not simplified versions