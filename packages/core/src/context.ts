import {
    Splitter,
    CodeChunk,
    AstCodeSplitter
} from './splitter';
import {
    Embedding,
    EmbeddingVector,
    OpenAIEmbedding
} from './embedding';
import {
    VectorDatabase,
    VectorDocument,
    VectorSearchResult,
    HybridSearchRequest,
    HybridSearchOptions,
    HybridSearchResult
} from './vectordb';
import { SemanticSearchResult } from './types';
import { envManager } from './utils/env-manager';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import { FileSynchronizer } from './sync/synchronizer';
import { fileExtensionToLanguage } from './language';

const DEFAULT_SUPPORTED_EXTENSIONS = [
    // Programming languages
    '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.m', '.mm',
    // Text and markup files
    '.md', '.markdown', '.ipynb',
    // '.txt',  '.json', '.yaml', '.yml', '.xml', '.html', '.htm',
    // '.css', '.scss', '.less', '.sql', '.sh', '.bash', '.env'
];

const DEFAULT_IGNORE_PATTERNS = [
    // Common build output and dependency directories
    'node_modules/**',
    'dist/**',
    'build/**',
    'out/**',
    'target/**',
    'coverage/**',
    '.nyc_output/**',

    // IDE and editor files
    '.vscode/**',
    '.idea/**',
    '*.swp',
    '*.swo',

    // Version control
    '.git/**',
    '.svn/**',
    '.hg/**',

    // Cache directories
    '.cache/**',
    '__pycache__/**',
    '.pytest_cache/**',

    // Logs and temporary files
    'logs/**',
    'tmp/**',
    'temp/**',
    '*.log',

    // Environment and config files
    '.env',
    '.env.*',
    '*.local',

    // Minified and bundled files
    '*.min.js',
    '*.min.css',
    '*.min.map',
    '*.bundle.js',
    '*.bundle.css',
    '*.chunk.js',
    '*.vendor.js',
    '*.polyfills.js',
    '*.runtime.js',
    '*.map', // source map files
    'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'out',
    'target', '.vscode', '.idea', '__pycache__', '.pytest_cache',
    'coverage', '.nyc_output', 'logs', 'tmp', 'temp'
];

export interface ContextConfig {
    embedding?: Embedding;
    vectorDatabase?: VectorDatabase;
    codeSplitter?: Splitter;
    supportedExtensions?: string[];
    ignorePatterns?: string[];
    customExtensions?: string[]; // New: custom extensions from MCP
    customIgnorePatterns?: string[]; // New: custom ignore patterns from MCP
    chunkSize?: number;
    chunkOverlap?: number;
    vectorDbType?: 'postgres' | 'milvus' | 'azure' | 'zilliz'; // New: specify vector database type
}

export class Context {
    private embedding: Embedding;
    private vectorDatabase: VectorDatabase;
    private codeSplitter: Splitter;
    private supportedExtensions: string[];
    private ignorePatterns: string[];
    private synchronizers = new Map<string, FileSynchronizer>();
    private vectorDbType: 'postgres' | 'milvus' | 'azure' | 'zilliz';

    constructor(config: ContextConfig = {}) {
        // Initialize services
        this.embedding = config.embedding || new OpenAIEmbedding({
            apiKey: envManager.get('OPENAI_API_KEY') || 'your-openai-api-key',
            model: envManager.get('EMBEDDING_MODEL') || 'text-embedding-3-small',
            ...(envManager.get('OPENAI_BASE_URL') && { baseURL: envManager.get('OPENAI_BASE_URL') })
        });

        if (!config.vectorDatabase) {
            throw new Error('VectorDatabase is required. Please provide a vectorDatabase instance in the config.');
        }
        this.vectorDatabase = config.vectorDatabase;

        // Detect vector database type from constructor name or config
        this.vectorDbType = config.vectorDbType || this.detectVectorDbType(config.vectorDatabase);
        console.log(`[Context] 🗄️  Using vector database type: ${this.vectorDbType}`);

        config.chunkSize = Number(envManager.get('INDEXING_CHUNK_SIZE') ?? 2500);
        config.chunkOverlap = Number(envManager.get('INDEXING_CHUNK_OVERLAP') ?? 300);

        this.codeSplitter = new AstCodeSplitter(config.chunkSize, config.chunkOverlap);

        // Load custom extensions from environment variables
        const envCustomExtensions = this.getCustomExtensionsFromEnv();

        // Combine default extensions with config extensions and env extensions
        const allSupportedExtensions = [
            ...DEFAULT_SUPPORTED_EXTENSIONS,
            ...(config.supportedExtensions || []),
            ...(config.customExtensions || []),
            ...envCustomExtensions
        ];
        // Remove duplicates
        this.supportedExtensions = [...new Set(allSupportedExtensions)];

        // Load custom ignore patterns from environment variables  
        const envCustomIgnorePatterns = this.getCustomIgnorePatternsFromEnv();

        // Start with default ignore patterns
        const allIgnorePatterns = [
            ...DEFAULT_IGNORE_PATTERNS,
            ...(config.ignorePatterns || []),
            ...(config.customIgnorePatterns || []),
            ...envCustomIgnorePatterns
        ];
        // Remove duplicates
        this.ignorePatterns = [...new Set(allIgnorePatterns)];

        console.log(`[Context] 🔧 Initialized with ${this.supportedExtensions.length} supported extensions and ${this.ignorePatterns.length} ignore patterns`);
        if (envCustomExtensions.length > 0) {
            console.log(`[Context] 📎 Loaded ${envCustomExtensions.length} custom extensions from environment: ${envCustomExtensions.join(', ')}`);
        }
        if (envCustomIgnorePatterns.length > 0) {
            console.log(`[Context] 🚫 Loaded ${envCustomIgnorePatterns.length} custom ignore patterns from environment: ${envCustomIgnorePatterns.join(', ')}`);
        }
    }

    /**
     * Detect vector database type from instance
     */
    private detectVectorDbType(vectorDb: VectorDatabase): 'postgres' | 'milvus' | 'azure' | 'zilliz' {
        const constructorName = vectorDb.constructor.name;

        if (constructorName.toLowerCase().includes('azure')) {
            return 'azure';
        } else if (constructorName.toLowerCase().includes('postgres')) {
            return 'postgres';
        } else if (constructorName.toLowerCase().includes('zilliz')) {
            return 'zilliz';
        } else if (constructorName.toLowerCase().includes('milvus')) {
            return 'milvus';
        }

        // Default to postgres if unable to detect
        console.warn(`[Context] ⚠️  Unable to detect vector database type from constructor name: ${constructorName}. Defaulting to postgres.`);
        return 'postgres';
    }

    /**
     * Get vector database type
     */
    getVectorDbType(): 'postgres' | 'milvus' | 'azure' | 'zilliz' {
        return this.vectorDbType;
    }

    /**
     * Get embedding instance
     */
    getEmbedding(): Embedding {
        return this.embedding;
    }

    /**
     * Get vector database instance
     */
    getVectorDatabase(): VectorDatabase {
        return this.vectorDatabase;
    }

    /**
     * Get code splitter instance
     */
    getCodeSplitter(): Splitter {
        return this.codeSplitter;
    }

    /**
     * Get supported extensions
     */
    getSupportedExtensions(): string[] {
        return [...this.supportedExtensions];
    }

    /**
     * Get ignore patterns
     */
    getIgnorePatterns(): string[] {
        return [...this.ignorePatterns];
    }

    /**
     * Get synchronizers map
     */
    getSynchronizers(): Map<string, FileSynchronizer> {
        return new Map(this.synchronizers);
    }

    /**
     * Set synchronizer for a collection
     */
    setSynchronizer(collectionName: string, synchronizer: FileSynchronizer): void {
        this.synchronizers.set(collectionName, synchronizer);
    }

    /**
     * Public wrapper for loadIgnorePatterns private method
     */
    async getLoadedIgnorePatterns(codebasePath: string): Promise<void> {
        return this.loadIgnorePatterns(codebasePath);
    }

    /**
     * Public wrapper for prepareCollection private method
     */
    async getPreparedCollection(codebasePath: string): Promise<void> {
        return this.prepareCollection(codebasePath);
    }

    /**
     * Get isHybrid setting from environment variable with default true
     */
    private getIsHybrid(): boolean {
        const isHybridEnv = envManager.get('HYBRID_MODE');
        if (isHybridEnv === undefined || isHybridEnv === null) {
            return true; // Default to true
        }
        return isHybridEnv.toLowerCase() === 'true';
    }

    /**
     * Generate collection name based on codebase path and hybrid mode
     * Azure AI Search requires specific naming conventions
     */
    public getCollectionName(codebasePath: string): string {
        const isHybrid = this.getIsHybrid();
        const normalizedPath = path.resolve(codebasePath);
        const hash = crypto.createHash('md5').update(normalizedPath).digest('hex');

        // Get base name from path
        const baseName = path.basename(normalizedPath);

        // Azure AI Search has strict naming requirements:
        // - Must be lowercase
        // - Can only contain letters, digits, or dashes
        // - Cannot start or end with dashes
        // - No consecutive dashes
        // - Max length is 128 characters
        if (this.vectorDbType === 'azure') {
            const azureSafeName = baseName
                .toLowerCase()
                .replace(/[^a-z0-9-]/g, '-')  // Replace non-alphanumeric with dash
                .replace(/--+/g, '-')          // Collapse consecutive dashes
                .replace(/^-|-$/g, '');        // Remove leading/trailing dashes

            const suffix = isHybrid ? '-hybrid' : '';
            const shortHash = hash.substring(0, 8);

            // Construct name ensuring it stays under 128 chars
            let collectionName = `${azureSafeName}${suffix}-${shortHash}`;

            // Truncate if necessary, ensuring we don't end with a dash
            if (collectionName.length > 128) {
                collectionName = collectionName.substring(0, 127).replace(/-$/, '');
            }

            console.log(`[Context] 🏷️  Azure AI Search collection name: ${collectionName}`);
            return collectionName;
        }

        // For other databases, use original logic
        const prefix = 'claude_context';
        const suffix = isHybrid ? '_hybrid' : '';
        return `${prefix}_${hash}${suffix}`;
    }

    /**
     * Convert filter expression to appropriate format for the vector database
     * Azure AI Search uses OData syntax, while others use different formats
     */
    private convertFilterExpression(filter: string | undefined): string | undefined {
        if (!filter) return undefined;

        if (this.vectorDbType === 'azure') {
            // Convert common filter patterns to OData syntax
            // Example: fileExtension = 'ts' -> fileExtension eq 'ts'
            // Example: startLine >= 10 -> startLine ge 10
            return filter
                .replace(/\s*=\s*/g, ' eq ')
                .replace(/\s*!=\s*/g, ' ne ')
                .replace(/\s*>=\s*/g, ' ge ')
                .replace(/\s*<=\s*/g, ' le ')
                .replace(/\s*>\s*/g, ' gt ')
                .replace(/\s*<\s*/g, ' lt ')
                .replace(/\s+AND\s+/gi, ' and ')
                .replace(/\s+OR\s+/gi, ' or ')
                .replace(/\s+NOT\s+/gi, ' not ');
        }

        // For PostgreSQL and Milvus, return as-is
        return filter;
    }

    /**
     * Build filter expression for queries
     * Handles different syntax requirements for different vector databases
     */
    public buildFilterExpression(conditions: Record<string, any>): string | undefined {
        if (!conditions || Object.keys(conditions).length === 0) {
            return undefined;
        }

        const clauses: string[] = [];

        for (const [key, value] of Object.entries(conditions)) {
            if (value === undefined || value === null) continue;

            if (this.vectorDbType === 'azure') {
                // Azure AI Search OData syntax
                if (typeof value === 'string') {
                    clauses.push(`${key} eq '${value}'`);
                } else if (typeof value === 'number') {
                    clauses.push(`${key} eq ${value}`);
                } else if (Array.isArray(value)) {
                    // IN clause: search.in(field, 'value1,value2', ',')
                    const values = value.map(v => typeof v === 'string' ? v : String(v)).join(',');
                    clauses.push(`search.in(${key}, '${values}', ',')`);
                }
            } else if (this.vectorDbType === 'postgres') {
                // PostgreSQL SQL syntax
                if (typeof value === 'string') {
                    clauses.push(`${key} = '${value}'`);
                } else if (typeof value === 'number') {
                    clauses.push(`${key} = ${value}`);
                } else if (Array.isArray(value)) {
                    const values = value.map(v => typeof v === 'string' ? `'${v}'` : v).join(', ');
                    clauses.push(`${key} IN (${values})`);
                }
            } else {
                // Milvus/Zilliz boolean expression syntax
                if (typeof value === 'string') {
                    clauses.push(`${key} == "${value}"`);
                } else if (typeof value === 'number') {
                    clauses.push(`${key} == ${value}`);
                } else if (Array.isArray(value)) {
                    const values = value.map(v => typeof v === 'string' ? `"${v}"` : v).join(', ');
                    clauses.push(`${key} in [${values}]`);
                }
            }
        }

        if (clauses.length === 0) return undefined;

        // Join clauses with appropriate operator
        const operator = this.vectorDbType === 'azure' ? ' and ' :
            this.vectorDbType === 'postgres' ? ' AND ' : ' && ';

        return clauses.join(operator);
    }

    /**
     * Prepare collection for indexing
     * Handles Azure AI Search specific requirements
     */
    private async prepareCollection(codebasePath: string): Promise<void> {
        const collectionName = this.getCollectionName(codebasePath);
        const isHybrid = this.getIsHybrid();

        try {
            const exists = await this.vectorDatabase.hasCollection(collectionName);

            if (!exists) {
                console.log(`[Context] 📦 Creating ${isHybrid ? 'hybrid' : 'standard'} collection: ${collectionName}`);

                // Get embedding dimension
                const dimension = await this.getEmbeddingDimension();

                if (isHybrid) {
                    await this.vectorDatabase.createHybridCollection(collectionName, dimension);
                } else {
                    await this.vectorDatabase.createCollection(collectionName, dimension);
                }

                if (this.vectorDbType === 'azure') {
                    console.log(`[Context] ✅ Azure AI Search index created: ${collectionName}`);
                } else {
                    console.log(`[Context] ✅ Collection created: ${collectionName}`);
                }
            } else {
                if (this.vectorDbType === 'azure') {
                    console.log(`[Context] ℹ️  Azure AI Search index already exists: ${collectionName}`);
                } else {
                    console.log(`[Context] ℹ️  Collection already exists: ${collectionName}`);
                }
            }
        } catch (error: any) {
            // Handle Azure-specific errors
            if (this.vectorDbType === 'azure' && error.message?.includes('collection limit')) {
                console.error(`[Context] ❌ Azure AI Search index limit reached. Please upgrade your tier or delete unused indexes.`);
                throw error;
            }
            throw error;
        }
    }

    /**
     * Get embedding dimension
     */
    private async getEmbeddingDimension(): Promise<number> {
        // Try to get from embedding model
        const modelName = envManager.get('EMBEDDING_MODEL') || 'text-embedding-3-small';

        // Common embedding dimensions
        const dimensionMap: Record<string, number> = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,
        };

        return dimensionMap[modelName] || 1536;
    }

    /**
     * Enhanced search with Azure AI Search optimizations
     */
    async search(
        codebasePath: string,
        query: string,
        options?: {
            topK?: number;
            threshold?: number;
            filter?: Record<string, any>;
            useHybrid?: boolean;
        }
    ): Promise<SemanticSearchResult[]> {
        const collectionName = this.getCollectionName(codebasePath);
        const topK = options?.topK || 10;
        const threshold = options?.threshold || 0.0;
        const useHybrid = options?.useHybrid !== false && this.getIsHybrid();

        // Generate query embedding
        const queryEmbedding = await this.embedding.embed(query);

        // Build filter expression
        const filterExpr = options?.filter ? this.buildFilterExpression(options.filter) : undefined;

        let results: VectorSearchResult[] | HybridSearchResult[];

        if (useHybrid && this.vectorDbType === 'azure') {
            // Azure AI Search native hybrid search
            console.log(`[Context] 🔍 Performing Azure AI Search hybrid search...`);

            const searchRequests: HybridSearchRequest[] = [
                {
                    data: queryEmbedding.vector,
                    anns_field: 'vector',
                    param: { metric_type: 'COSINE' },
                    limit: topK * 2
                },
                {
                    data: query,
                    anns_field: 'sparse_vector',
                    param: {},
                    limit: topK * 2
                }
            ];

            results = await this.vectorDatabase.hybridSearch(
                collectionName,
                searchRequests,
                {
                    limit: topK,
                    filterExpr,
                    rerank: {
                        strategy: 'weighted',
                        params: { weights: [0.7, 0.3] }
                    }
                }
            );
        } else {
            // Standard vector search
            console.log(`[Context] 🔍 Performing vector search...`);

            results = await this.vectorDatabase.search(
                collectionName,
                queryEmbedding.vector,
                {
                    topK,
                    threshold,
                    filterExpr
                }
            );
        }

        // Convert to SemanticSearchResult format
        return results.map(result => ({

            content: result.document.content,
            relativePath: result.document.relativePath,
            startLine: result.document.startLine,
            endLine: result.document.endLine,
            language: fileExtensionToLanguage(result.document.fileExtension),

            score: result.score,
            metadata: result.document.metadata
        }));
    }


    /**
     * Load ignore patterns from various sources
     */
    private async loadIgnorePatterns(codebasePath: string): Promise<void> {
        // Load from .contextignore files
        const localPatterns = await this.loadLocalIgnoreFile(codebasePath);
        const globalPatterns = await this.loadGlobalIgnoreFile();

        // Merge patterns
        const allPatterns = [
            ...this.ignorePatterns,
            ...localPatterns,
            ...globalPatterns
        ];

        // Remove duplicates
        this.ignorePatterns = [...new Set(allPatterns)];
    }

    /**
     * Load local .contextignore file from codebase root
     * @param codebasePath Path to the codebase
     * @returns Array of ignore patterns
     */
    private async loadLocalIgnoreFile(codebasePath: string): Promise<string[]> {
        try {
            const localIgnorePath = path.join(codebasePath, '.contextignore');
            return await this.loadIgnoreFile(localIgnorePath, 'local .contextignore');
        } catch (error) {
            // Local ignore file is optional
            return [];
        }
    }

    /**
     * Load global ignore file from ~/.context/.contextignore
     * @returns Array of ignore patterns
     */
    private async loadGlobalIgnoreFile(): Promise<string[]> {
        try {
            const homeDir = require('os').homedir();
            const globalIgnorePath = path.join(homeDir, '.context', '.contextignore');
            return await this.loadIgnoreFile(globalIgnorePath, 'global .contextignore');
        } catch (error) {
            // Global ignore file is optional, don't log warnings
            return [];
        }
    }

    /**
     * Load ignore patterns from a specific ignore file
     * @param filePath Path to the ignore file
     * @param fileName Display name for logging
     * @returns Array of ignore patterns
     */
    private async loadIgnoreFile(filePath: string, fileName: string): Promise<string[]> {
        try {
            await fs.promises.access(filePath);
            console.log(`📄 Found ${fileName} file at: ${filePath}`);

            const ignorePatterns = await Context.getIgnorePatternsFromFile(filePath);

            if (ignorePatterns.length > 0) {
                console.log(`[Context] 🚫 Loaded ${ignorePatterns.length} ignore patterns from ${fileName}`);
                return ignorePatterns;
            } else {
                console.log(`📄 ${fileName} file found but no valid patterns detected`);
                return [];
            }
        } catch (error) {
            if (fileName.includes('global')) {
                console.log(`📄 No ${fileName} file found`);
            }
            return [];
        }
    }

    /**
     * Static method to read ignore patterns from file
     */
    private static async getIgnorePatternsFromFile(filePath: string): Promise<string[]> {
        try {
            const content = await fs.promises.readFile(filePath, 'utf-8');
            return content
                .split('\n')
                .map(line => line.trim())
                .filter(line => line && !line.startsWith('#'));
        } catch (error) {
            return [];
        }
    }

    /**
     * Check if a path matches any ignore pattern
     * @param filePath Path to check
     * @param basePath Base path for relative pattern matching
     * @returns True if path should be ignored
     */
    private matchesIgnorePattern(filePath: string, basePath: string): boolean {
        if (this.ignorePatterns.length === 0) {
            return false;
        }

        const relativePath = path.relative(basePath, filePath);
        const normalizedPath = relativePath.replace(/\\/g, '/'); // Normalize path separators

        for (const pattern of this.ignorePatterns) {
            if (this.isPatternMatch(normalizedPath, pattern)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Simple glob pattern matching
     * @param filePath File path to test
     * @param pattern Glob pattern
     * @returns True if pattern matches
     */
    private isPatternMatch(filePath: string, pattern: string): boolean {
        // Handle directory patterns (ending with /)
        if (pattern.endsWith('/')) {
            const dirPattern = pattern.slice(0, -1);
            const pathParts = filePath.split('/');
            return pathParts.some(part => this.simpleGlobMatch(part, dirPattern));
        }

        // Handle file patterns
        if (pattern.includes('/')) {
            // Pattern with path separator - match exact path
            return this.simpleGlobMatch(filePath, pattern);
        } else {
            // Pattern without path separator - match filename in any directory
            const fileName = path.basename(filePath);
            return this.simpleGlobMatch(fileName, pattern);
        }
    }

    /**
     * Simple glob matching supporting * wildcard
     * @param text Text to test
     * @param pattern Pattern with * wildcards
     * @returns True if pattern matches
     */
    private simpleGlobMatch(text: string, pattern: string): boolean {
        // Convert glob pattern to regex
        const regexPattern = pattern
            .replace(/[.+^${}()|[\]\\]/g, '\\$&') // Escape regex special chars except *
            .replace(/\*/g, '.*'); // Convert * to .*

        const regex = new RegExp(`^${regexPattern}$`);
        return regex.test(text);
    }

    /**
     * Get custom extensions from environment variables
     * Supports CUSTOM_EXTENSIONS as comma-separated list
     * @returns Array of custom extensions
     */
    private getCustomExtensionsFromEnv(): string[] {
        const envExtensions = envManager.get('CUSTOM_EXTENSIONS');
        if (!envExtensions) {
            return [];
        }

        try {
            const extensions = envExtensions
                .split(',')
                .map(ext => ext.trim())
                .filter(ext => ext.length > 0)
                .map(ext => ext.startsWith('.') ? ext : `.${ext}`); // Ensure extensions start with dot

            return extensions;
        } catch (error) {
            console.warn(`[Context] ⚠️  Failed to parse CUSTOM_EXTENSIONS: ${error}`);
            return [];
        }
    }

    /**
     * Get custom ignore patterns from environment variables  
     * Supports CUSTOM_IGNORE_PATTERNS as comma-separated list
     * @returns Array of custom ignore patterns
     */
    private getCustomIgnorePatternsFromEnv(): string[] {
        const envIgnorePatterns = envManager.get('CUSTOM_IGNORE_PATTERNS');
        if (!envIgnorePatterns) {
            return [];
        }

        try {
            const patterns = envIgnorePatterns
                .split(',')
                .map(pattern => pattern.trim())
                .filter(pattern => pattern.length > 0);

            return patterns;
        } catch (error) {
            console.warn(`[Context] ⚠️  Failed to parse CUSTOM_IGNORE_PATTERNS: ${error}`);
            return [];
        }
    }

    /**
     * Add custom extensions (from MCP or other sources) without replacing existing ones
     * @param customExtensions Array of custom extensions to add
     */
    addCustomExtensions(customExtensions: string[]): void {
        if (customExtensions.length === 0) return;

        // Ensure extensions start with dot
        const normalizedExtensions = customExtensions.map(ext =>
            ext.startsWith('.') ? ext : `.${ext}`
        );

        // Merge current extensions with new custom extensions, avoiding duplicates
        const mergedExtensions = [...this.supportedExtensions, ...normalizedExtensions];
        const uniqueExtensions: string[] = [...new Set(mergedExtensions)];
        this.supportedExtensions = uniqueExtensions;
        console.log(`[Context] 📎 Added ${customExtensions.length} custom extensions. Total: ${this.supportedExtensions.length} extensions`);
    }

    /**
     * Get current splitter information
     */
    getSplitterInfo(): { type: string; hasBuiltinFallback: boolean; supportedLanguages?: string[] } {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
            return {
                type: 'ast',
                hasBuiltinFallback: true,
                supportedLanguages: AstCodeSplitter.getSupportedLanguages()
            };
        } else {
            return {
                type: 'langchain',
                hasBuiltinFallback: false
            };
        }
    }

    /**
     * Check if current splitter supports a specific language
     * @param language Programming language
     */
    isLanguageSupported(language: string): boolean {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
            return AstCodeSplitter.isLanguageSupported(language);
        }

        // LangChain splitter supports most languages
        return true;
    }

    /**
     * Get which strategy would be used for a specific language
     * @param language Programming language
     */
    getSplitterStrategyForLanguage(language: string): { strategy: 'ast' | 'langchain'; reason: string } {
        const splitterName = this.codeSplitter.constructor.name;

        if (splitterName === 'AstCodeSplitter') {
            const { AstCodeSplitter } = require('./splitter/ast-splitter');
            const isSupported = AstCodeSplitter.isLanguageSupported(language);

            return {
                strategy: isSupported ? 'ast' : 'langchain',
                reason: isSupported
                    ? 'Language supported by AST parser'
                    : 'Language not supported by AST, will fallback to LangChain'
            };
        } else {
            return {
                strategy: 'langchain',
                reason: 'Using LangChain splitter directly'
            };
        }
    }

    /**
     * Get database-specific information for diagnostics
     */
    getDatabaseInfo(): {
        type: 'postgres' | 'milvus' | 'azure' | 'zilliz';
        features: string[];
        limitations: string[];
    } {
        const features: string[] = [];
        const limitations: string[] = [];

        switch (this.vectorDbType) {
            case 'azure':
                features.push(
                    'Native hybrid search (vector + full-text)',
                    'Automatic scaling',
                    'Built-in semantic ranking',
                    'OData filter expressions',
                    'Fully managed service'
                );
                limitations.push(
                    'Collection naming restrictions (lowercase, alphanumeric, dashes only)',
                    'Maximum 128 characters for collection names',
                    'Tier-based collection limits',
                    'No GPU acceleration'
                );
                break;
            case 'postgres':
                features.push(
                    'Full SQL capabilities',
                    'Rich ecosystem of tools',
                    'ACID compliance',
                    'Flexible filtering',
                    'Open source'
                );
                limitations.push(
                    'Manual scaling required',
                    'Performance degrades with large datasets',
                    'Limited native vector operations'
                );
                break;
            case 'milvus':
            case 'zilliz':
                features.push(
                    'Purpose-built for vectors',
                    'Excellent performance at scale',
                    'GPU acceleration support',
                    'Multiple index types',
                    'Horizontal scaling'
                );
                limitations.push(
                    'Additional infrastructure component',
                    'Limited full-text search',
                    'Learning curve for operations'
                );
                break;
        }

        return {
            type: this.vectorDbType,
            features,
            limitations
        };
    }
}